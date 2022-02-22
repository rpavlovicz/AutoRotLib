#!/Users/rpavlovicz/anaconda3/envs/oepython3/bin/python

import sys
import argparse
from math import degrees
from openeye.oechem import *
from openeye.oeomega import *
from openeye.oequacpac import *
from openeye.oeszybki import *

def check_arg(args=None):
  parser = argparse.ArgumentParser(description="create N input files for MakeRotLib, where N = n_phi_bins*n_psi_bins")
  parser.add_argument('-n', '--aa_name', required='True', type=str, help='three letter name for residue')
  parser.add_argument('--charge', required='True', type=int, help='total charge of side chain')
  parser.add_argument('--smiles', required='True', type=str, help='input smiles string (with acetyl N-term extention and N-methyl C-term extension')
  results = parser.parse_args(args)
  return (results.smiles, results.aa_name, results.charge)

def main(argv=[__name__]):

  infile, aa_name, sidechain_charge = check_arg(sys.argv[1:])
  
  ### make sure input aa_name is three characters long
  if len(aa_name) != 3:
      sys.exit('aa_name should be three characters long')

  ifs = oemolistream()
  if not ifs.open(infile):
    OEThrow.Fatal("Unable to open %s for reading" % infile)

  outfile = '%s_AM1BCC.mol2'%(aa_name)
  ofs = oemolostream()
  if not ofs.open(outfile):
    OEThrow.Fatal("Unable to open %s for writing" % outfile)
    OEThrow.Usage("%s <infile> <outfile>" % argv[0])

  # run in verbose mode to get torlib info
  oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Verbose)

  # set torlib 
  torLib = OETorLib()
  torLib.SetTorsionLibrary(OETorLibType_Original) # options = OETorLibType_Original (OE-version) or OETorLibType_GubaV21

  omegaOpts = OEOmegaOptions()
  omegaOpts.SetSampleHydrogens(True) # override default of fixed hydrogens
  omegaOpts.SetRMSThreshold(0.2)
  omegaOpts.SetMaxConfs(500)
  omega = OEOmega(omegaOpts)
  omega.SetCommentEnergy(True)  # write energy in kcal/mol to comment field of molecule
  omega.SetWarts(True)  # generate_unique titles for conformers
  omegaOpts.SetTorLib(torLib)

  omega.SetOptions(omegaOpts)

  PEPTOID = False
  NMETH = False
  NCYCLE = False
  CACYCLE = False
  CABRANCH = False

  for mol in ifs.GetOEMols():
    if omega(mol):
      nconfs = mol.GetMaxConfIdx()
      print('\n***\ngenerated %i conformers for %s\n***'%(nconfs,infile))
      print('assigning AM1BCC charges with ELF10 method...')
      OEAssignCharges(mol, OEAM1BCCELF10Charges())
      OETriposAtomTypeNames(mol)  # this normally happens when writing with OEWriteMolecule
      OETriposAtomNames(mol)      # this normally happens when writing with OEWriteMolecule

      # identify backbone atoms
      ss = OESubSearch("CC(=O)NCC(=O)NC")  # N-to-C
      OEPrepareSearch(mol, ss)

      if ss.SingleMatch(mol):
        print('\nfound single backbone match in query')
      else:
        sys.exit('single backbone match not found in input')

      bb_list = []; bb_charge_sum = 0; bb_charge_sum_uncapped = 0;
      ignore_list = []
      bb_idx_list = []  # this is dumb, but used for summing uncapped backbone charges
      ha_list = []
      for match in ss.Match(mol):
        for ma in match.GetAtoms():
          ### store backbone atoms
          if ma.pattern.GetIdx() == 1: lower = ma.target
          elif ma.pattern.GetIdx() == 3: N = ma.target; bb_idx_list.append(ma.target.GetIdx())
          elif ma.pattern.GetIdx() == 4: CA = ma.target; bb_idx_list.append(ma.target.GetIdx())
          elif ma.pattern.GetIdx() == 5: C = ma.target; bb_idx_list.append(ma.target.GetIdx())
          elif ma.pattern.GetIdx() == 6: O = ma.target; bb_idx_list.append(ma.target.GetIdx())
          elif ma.pattern.GetIdx() == 7: upper = ma.target
          else: ignore_list.append(ma.target.GetIdx())

          bb_list.append(ma.target.GetIdx())
          bb_charge_sum += ma.target.GetPartialCharge()
          if ma.pattern.GetIdx() <= 6 and ma.pattern.GetIdx() >= 3:
            bb_charge_sum_uncapped += ma.target.GetPartialCharge()

          ### find backbone protons
          for nbr in ma.target.GetAtoms():
            if nbr.GetType() == 'H':
              bb_list.append(nbr.GetIdx())
              bb_charge_sum += nbr.GetPartialCharge()
              if ma.pattern.GetIdx() <= 6 and ma.pattern.GetIdx() >= 3:
                bb_charge_sum_uncapped += nbr.GetPartialCharge()
              if ma.pattern.GetIdx() == 3:
                H = nbr
                bb_idx_list.append(nbr.GetIdx())
              elif ma.pattern.GetIdx() == 4:
                ha_list.append(nbr)
              elif ma.pattern.GetIdx() == 7:
                ignore_list.append(nbr.GetIdx())
              elif ma.target.GetIdx() in ignore_list:
                ignore_list.append(nbr.GetIdx())

          if ma.pattern.GetIdx() == 4:
            CA = ma.target  # save CA atom object

      if len(ha_list) > 2:
        sys.exit('ERROR: detected more than two protons bound to backbone CA?!')

      ### check here for CACYCLE and CABRANCH
      if len(ha_list) == 0:
        bound_to_CA = []
        for bond in CA.GetBonds():
          if bond.GetBgn() != CA and bond.GetBgn().GetIdx() not in bb_list:
            bound_to_CA.append(bond.GetBgn())
          if bond.GetEnd() != CA and bond.GetEnd().GetIdx() not in bb_list:
              bound_to_CA.append(bond.GetEnd())
        print('number of non-backbone heavy atoms bound to CA = %i'%(len(bound_to_CA)))
        if len(bound_to_CA) == 2:
          if bound_to_CA[0].IsInRing() and bound_to_CA[1].IsInRing():
            CACYCLE = True
          elif not bound_to_CA[0].IsInRing() and not bound_to_CA[1].IsInRing():
            CABRANCH = True

      ### check here for peptoid / N-methyl / backbone-cyclization
      bound_to_N = []
      for bond in N.GetBonds():
        if bond.GetBgn() != N and bond.GetBgn().GetIdx() not in bb_list:
          bound_to_N.append(bond.GetBgn())
        elif bond.GetEnd() != N and bond.GetEnd().GetIdx() not in bb_list:
          bound_to_N.append(bond.GetEnd())
      if len(bound_to_N) > 1:
        sys.exit('ERROR: identified more than one non-backbone atom bound to backbone N')
      elif len(bound_to_N) == 1:
        if bound_to_N[0].IsInRing():
          ### TODO: check that the ring includes CA atom
          print('detected backbone cyclization to N atom')
          NCYCLE = True
        ### N-methyl and peptoid detection
        else:
          non_H_NC = []
          H_NC = []
          for atom in bound_to_N[0].GetAtoms():
            if (atom != N) and (atom.GetType() != 'H'):
              non_H_NC.append(atom)
            elif (atom != N) and (atom.GetType() == 'H'):
              H_NC.append(atom)
          print('*** length of non_H_NC = %i'%(len(non_H_NC)))
          ### allow both NMETH + CACYCLE or CABRANCH
          if (len(non_H_NC) == 0 and len(H_NC) == 3):
            NMETH = True
            CM = bound_to_N[0]
            HM1 = H_NC[0]
            HM2 = H_NC[1]
            HM3 = H_NC[2]
            if len(ha_list) == 1:
              HA = ha_list[0]
              bb_list.append(HA.GetIdx()); bb_idx_list.append(HA.GetIdx())
              ### just to be safe
              if (CACYCLE == True) or (CABRANCH == True):
                sys.exit('error: detected both CA_BRANCHING/CYCLIZTION and a proton on the CA atom')
            ### special case for N-methyl glycine
            elif len(ha_list) == 2:
              HA1 = ha_list[0]
              HA2 = ha_list[1]
              bb_list.append(HA1.GetIdx()); bb_idx_list.append(HA1.GetIdx())
              bb_list.append(HA2.GetIdx()); bb_idx_list.append(HA2.GetIdx())

            ### set N-methyl group to backbone
            bb_list.append(CM.GetIdx()); bb_idx_list.append(CM.GetIdx())
            bb_list.append(HM1.GetIdx()); bb_idx_list.append(HM1.GetIdx())
            bb_list.append(HM2.GetIdx()); bb_idx_list.append(HM2.GetIdx())
            bb_list.append(HM3.GetIdx()); bb_idx_list.append(HM3.GetIdx())
            print('detected N-methyl backbone')
          ### otherwise should be peptoid
          elif (len(non_H_NC) >= 0 and len(ha_list) == 2):
            PEPTOID = True
            HA1 = ha_list[0]
            HA2 = ha_list[1]
            bb_list.append(HA1.GetIdx()); bb_idx_list.append(HA1.GetIdx())
            bb_list.append(HA2.GetIdx()); bb_idx_list.append(HA2.GetIdx())
            print('petoid detected')

      print('\nTotal initial charge for backbone = %f'%(bb_charge_sum))
      print('Total initial charge for backbone (excluding caps) = %f\n'%(bb_charge_sum_uncapped))

      if (PEPTOID == False) and (NMETH == False) and (CACYCLE == False) and (CABRANCH == False):
        HA = ha_list[0]
        bb_list.append(HA.GetIdx()); bb_idx_list.append(HA.GetIdx())

      print('NCYCLE = %s'%(NCYCLE))
      print('CACYCLE = %s'%(CACYCLE))
      print('CABRANCH = %s'%(CABRANCH))
      print('PEPTOID = %s'%(PEPTOID))
      print('NMETH = %s'%(NMETH))
      # adjust backbone partial charges for Talaris compatibility
      if NCYCLE:
        print('\nproline-like REF2015 backbone charges:')
        print('  partial charge on  N being modified from %7.4f to -0.3730 -- difference = %7.4f'%(N.GetPartialCharge(),-0.3730-N.GetPartialCharge()))
        print('  partial charge on CA being modified from %7.4f to  0.0257 -- difference = %7.4f'%(CA.GetPartialCharge(),0.0257-CA.GetPartialCharge()))
        print('  partial charge on HA being modified from %7.4f to  0.1158 -- difference = %7.4f'%(HA.GetPartialCharge(),0.1158-HA.GetPartialCharge()))
        print('  partial charge on  C being modified from %7.4f to  0.6885 -- difference = %7.4f'%(C.GetPartialCharge(),0.6885-C.GetPartialCharge()))
        print('  partial charge on  O being modified from %7.4f to -0.6885 -- difference = %7.4f'%(O.GetPartialCharge(),-0.6885-O.GetPartialCharge()))
        N.SetPartialCharge( -0.3730 )
        CA.SetPartialCharge( 0.0257 )
        HA.SetPartialCharge( 0.1158 )
        C.SetPartialCharge( 0.6885 )
        O.SetPartialCharge( -0.6885 )
      elif ( NMETH and CABRANCH ):
        print('\n assigning N-methyl + CA-branched backbone charges:')
        print('  partial charge on   N being modified from %7.4f to -0.3242 -- difference = %7.4f'%(N.GetPartialCharge(),-0.3242-N.GetPartialCharge()))
        print('  partial charge on  CM being modified from %7.4f to -0.0999 -- difference = %7.4f'%(CM.GetPartialCharge(),-0.0999-CM.GetPartialCharge()))
        print('  partial charge on HM1 being modified from %7.4f to  0.0333 -- difference = %7.4f'%(HM1.GetPartialCharge(),0.0333-HM1.GetPartialCharge()))
        print('  partial charge on HM2 being modified from %7.4f to  0.0333 -- difference = %7.4f'%(HM2.GetPartialCharge(),0.0333-HM2.GetPartialCharge()))
        print('  partial charge on HM3 being modified from %7.4f to  0.0333 -- difference = %7.4f'%(HM3.GetPartialCharge(),0.0333-HM3.GetPartialCharge()))
        print('  partial charge on  CA being modified from %7.4f to  0.1500 -- difference = %7.4f'%(CA.GetPartialCharge(),0.1500-CA.GetPartialCharge()))
        print('  partial charge on   C being modified from %7.4f to  0.6204 -- difference = %7.4f'%(C.GetPartialCharge(),0.6204-C.GetPartialCharge()))
        print('  partial charge on   O being modified from %7.4f to -0.6204 -- difference = %7.4f'%(O.GetPartialCharge(),-0.6204-O.GetPartialCharge()))
        N.SetPartialCharge(  -0.3242 )
        CM.SetPartialCharge( -0.0999 )
        HM1.SetPartialCharge( 0.0333 )
        HM2.SetPartialCharge( 0.0333 )
        HM3.SetPartialCharge( 0.0333 )
        CA.SetPartialCharge(  0.1500 )
        C.SetPartialCharge(   0.6204 )
        O.SetPartialCharge(  -0.6204 )
      ### special case for N-methyl glycine
      elif ( NMETH and len(ha_list) == 2 ):
        print('\nassigning N-methyl glycine backbone charges:')
        print('  partial charge on   N being modified from %7.4f to -0.3242 -- difference = %7.4f'%(N.GetPartialCharge(),-0.3242-N.GetPartialCharge()))
        print('  partial charge on  CM being modified from %7.4f to -0.0999 -- difference = %7.4f'%(CM.GetPartialCharge(),-0.0999-CM.GetPartialCharge()))
        print('  partial charge on HM1 being modified from %7.4f to  0.0333 -- difference = %7.4f'%(HM1.GetPartialCharge(),0.0333-HM1.GetPartialCharge()))
        print('  partial charge on HM2 being modified from %7.4f to  0.0333 -- difference = %7.4f'%(HM2.GetPartialCharge(),0.0333-HM2.GetPartialCharge()))
        print('  partial charge on HM3 being modified from %7.4f to  0.0333 -- difference = %7.4f'%(HM3.GetPartialCharge(),0.0333-HM3.GetPartialCharge()))
        print('  partial charge on  CA being modified from %7.4f to  0.0608 -- difference = %7.4f'%(CA.GetPartialCharge(),0.0608-CA.GetPartialCharge()))
        print('  partial charge on HA1 being modified from %7.4f to  0.1317 -- difference = %7.4f'%(HA1.GetPartialCharge(),0.1317-HA1.GetPartialCharge()))
        print('  partial charge on HA2 being modified from %7.4f to  0.1317 -- difference = %7.4f'%(HA2.GetPartialCharge(),0.1317-HA2.GetPartialCharge()))
        print('  partial charge on   C being modified from %7.4f to  0.6195 -- difference = %7.4f'%(C.GetPartialCharge(),0.6195-C.GetPartialCharge()))
        print('  partial charge on   O being modified from %7.4f to -0.6195 -- difference = %7.4f'%(O.GetPartialCharge(),-0.6195-O.GetPartialCharge()))
        N.SetPartialCharge(  -0.3242 )
        CM.SetPartialCharge( -0.0999 )
        HM1.SetPartialCharge( 0.0333 )
        HM2.SetPartialCharge( 0.0333 )
        HM3.SetPartialCharge( 0.0333 )
        CA.SetPartialCharge(  0.0608 )
        HA1.SetPartialCharge( 0.1317 )
        HA2.SetPartialCharge( 0.1317 )
        C.SetPartialCharge(   0.6195 )
        O.SetPartialCharge(  -0.6195 )
      elif NMETH:
        print('\nassigning N-methyl backbone charges:')
        print('  partial charge on   N being modified from %7.4f to -0.3242 -- difference = %7.4f'%(N.GetPartialCharge(),-0.3242-N.GetPartialCharge()))
        print('  partial charge on  CM being modified from %7.4f to -0.0999 -- difference = %7.4f'%(CM.GetPartialCharge(),-0.0999-CM.GetPartialCharge()))
        print('  partial charge on HM1 being modified from %7.4f to  0.0333 -- difference = %7.4f'%(HM1.GetPartialCharge(),0.0333-HM1.GetPartialCharge()))
        print('  partial charge on HM2 being modified from %7.4f to  0.0333 -- difference = %7.4f'%(HM2.GetPartialCharge(),0.0333-HM2.GetPartialCharge()))
        print('  partial charge on HM3 being modified from %7.4f to  0.0333 -- difference = %7.4f'%(HM3.GetPartialCharge(),0.0333-HM3.GetPartialCharge()))
        print('  partial charge on  CA being modified from %7.4f to  0.1968 -- difference = %7.4f'%(CA.GetPartialCharge(),0.1968-CA.GetPartialCharge()))
        print('  partial charge on  HA being modified from %7.4f to  0.1274 -- difference = %7.4f'%(HA.GetPartialCharge(),0.1274-HA.GetPartialCharge()))
        print('  partial charge on   C being modified from %7.4f to  0.6195 -- difference = %7.4f'%(C.GetPartialCharge(),0.6195-C.GetPartialCharge()))
        print('  partial charge on   O being modified from %7.4f to -0.6195 -- difference = %7.4f'%(O.GetPartialCharge(),-0.6195-O.GetPartialCharge()))
        N.SetPartialCharge(  -0.3242 )
        CM.SetPartialCharge( -0.0999 )
        HM1.SetPartialCharge( 0.0333 )
        HM2.SetPartialCharge( 0.0333 )
        HM3.SetPartialCharge( 0.0333 )
        CA.SetPartialCharge(  0.1968 )
        HA.SetPartialCharge( 0.1274 )
        C.SetPartialCharge(   0.6195 )
        O.SetPartialCharge(  -0.6195 )
      elif PEPTOID:
        print('\nassigning peptoid backbone charges:')
        print('  partial charge on   N being modified from %7.4f to -0.3242 -- difference = %7.4f'%(N.GetPartialCharge(),-0.3242-N.GetPartialCharge()))
        print('  partial charge on  CA being modified from %7.4f to  0.0608 -- difference = %7.4f'%(CA.GetPartialCharge(),0.0608-CA.GetPartialCharge()))
        print('  partial charge on HA1 being modified from %7.4f to  0.1317 -- difference = %7.4f'%(HA1.GetPartialCharge(),0.1317-HA1.GetPartialCharge()))
        print('  partial charge on HA2 being modified from %7.4f to  0.1317 -- difference = %7.4f'%(HA2.GetPartialCharge(),0.1317-HA2.GetPartialCharge()))
        print('  partial charge on   C being modified from %7.4f to  0.6081 -- difference = %7.4f'%(C.GetPartialCharge(),0.6081-C.GetPartialCharge()))
        print('  partial charge on   O being modified from %7.4f to -0.6081 -- difference = %7.4f'%(O.GetPartialCharge(),-0.6081-O.GetPartialCharge()))
        N.SetPartialCharge(  -0.3242 )
        CA.SetPartialCharge(  0.0608 )
        HA1.SetPartialCharge( 0.1317 )
        HA2.SetPartialCharge( 0.1317 )
        C.SetPartialCharge(   0.6081 )
        O.SetPartialCharge(  -0.6081 )
      elif ( CACYCLE or CABRANCH ):
        print('\nassigning ca-branched backbone charges:')
        print('  partial charge on   N being modified from %7.4f to -0.5200 -- difference = %7.4f'%(N.GetPartialCharge(),-0.5200-N.GetPartialCharge()))
        print('  partial charge on   H being modified from %7.4f to  0.3900 -- difference = %7.4f'%(H.GetPartialCharge(),-0.3900-H.GetPartialCharge()))
        print('  partial charge on  CA being modified from %7.4f to  0.1300 -- difference = %7.4f'%(CA.GetPartialCharge(),0.1300-CA.GetPartialCharge()))
        print('  partial charge on   C being modified from %7.4f to  0.6284 -- difference = %7.4f'%(C.GetPartialCharge(),0.6284-C.GetPartialCharge()))
        print('  partial charge on   O being modified from %7.4f to -0.6284 -- difference = %7.4f'%(O.GetPartialCharge(),-0.6284-O.GetPartialCharge()))
        N.SetPartialCharge(  -0.5200 )
        H.SetPartialCharge(   0.3900 )
        CA.SetPartialCharge(  0.1300 )
        C.SetPartialCharge(   0.6284 )
        O.SetPartialCharge(  -0.6284 )
      else:
        print('\nstandard REF2015 backbone charges:')
        print('  partial charge on  N being modified from %7.4f to -0.6046 -- difference = %7.4f'%(N.GetPartialCharge(),-0.6046-N.GetPartialCharge()))
        print('  partial charge on  H being modified from %7.4f to  0.3988 -- difference = %7.4f'%(H.GetPartialCharge(),0.3988-H.GetPartialCharge()))
        print('  partial charge on CA being modified from %7.4f to  0.0900 -- difference = %7.4f'%(CA.GetPartialCharge(),0.0900-CA.GetPartialCharge()))
        print('  partial charge on HA being modified from %7.4f to  0.1158 -- difference = %7.4f'%(HA.GetPartialCharge(),0.1158-HA.GetPartialCharge()))
        print('  partial charge on  C being modified from %7.4f to  0.6885 -- difference = %7.4f'%(C.GetPartialCharge(),0.6885-C.GetPartialCharge()))
        print('  partial charge on  O being modified from %7.4f to -0.6885 -- difference = %7.4f'%(O.GetPartialCharge(),-0.6885-O.GetPartialCharge()))
        N.SetPartialCharge( -0.6046 )
        H.SetPartialCharge(  0.3988 )
        CA.SetPartialCharge( 0.0900 )
        HA.SetPartialCharge( 0.1158 )
        C.SetPartialCharge(  0.6885 )
        O.SetPartialCharge( -0.6885 )

      ### rescan atoms and sum charge for backbone after partial charge reassignment
      fixed_bb_sum = 0
      for atom in mol.GetAtoms():
        if (atom.GetIdx() in bb_idx_list) and (atom.GetIdx() not in ignore_list):
          fixed_bb_sum += atom.GetPartialCharge()
      print('adjusted backbone charge sum = %f\n'%(fixed_bb_sum))

      sc_charge_sum = 0
      for atom in mol.GetAtoms():
        if atom.GetIdx() not in bb_list:
          sc_charge_sum += atom.GetPartialCharge()
      print('Total charge for side chain = %f'%(sc_charge_sum))

      ### perform charge correction between sc_charge_sum and fixed_bb_sum
      if NCYCLE:
        ### find CA and N substituents to split charge correction
        n_ca_bonds = 0; CA_list = []
        for x in CA.GetAtoms():
          n_ca_bonds += 1
          if x.GetIdx() not in bb_list:
            CA_list.append(x)

        n_n_bonds = 0; N_list = []
        for x in N.GetAtoms():
          n_n_bonds += 1
          if x.GetIdx() not in bb_list:
            N_list.append(x)

        if (len(CA_list) != 1) and (len(N_list) != 1):
          sys.exit('ERROR: found unexpected number of substituents on CA ( %i ) and/or N ( %i ) atoms'%(len(CA_list),len(N_list)))
        else:
          ### split the excess charge from input formal charge to computed partial charges from AM1BCC ELF10 protocol
          charge_correct = (sidechain_charge - sc_charge_sum - fixed_bb_sum)/2.0

        ### now split the charge
        for atom in CA_list:
          current_charge = atom.GetPartialCharge()
          atom.SetPartialCharge( current_charge + charge_correct )
          print('  adjusting charge of atom %s by %f: %f -> %f'%(atom.GetName(),charge_correct,current_charge,atom.GetPartialCharge()))

        for atom in N_list:
          current_charge = atom.GetPartialCharge()
          atom.SetPartialCharge( current_charge + charge_correct )
          print('  adjusting charge of atom %s by %f: %f -> %f'%(atom.GetName(),charge_correct,current_charge,atom.GetPartialCharge()))

      elif PEPTOID:
        ### perform charge correction between N and CA
        n_n_bonds = 0; N_list = []
        for x in N.GetAtoms():
          n_n_bonds += 1
          if x.GetIdx() not in bb_list:
            N_list.append(x)

        if (len(N_list) != 1):
          sys.exit('ERROR: found more %i non-backbone N substituent. There should only be 1 for a peptoid.'%(len(N_list)))
        else:
          charge_correct = (sidechain_charge - sc_charge_sum - fixed_bb_sum)

        for atom in N_list:
          current_charge = atom.GetPartialCharge()
          atom.SetPartialCharge( current_charge + charge_correct )
          print('  adjusting charge of atom %s by %f: %f -> %f'%(atom.GetName(),charge_correct,current_charge,atom.GetPartialCharge()))

      elif NMETH and ( len(ha_list) == 2 ):
        ### special case of N-methyl glycine (no charge correction)
        print('  no charge correction for N-methyl glycine') 

      else:
        ### alpha-amino acid including NMETH

        n_ca_bonds = 0; CB_list = []
        for x in CA.GetAtoms():
          n_ca_bonds += 1
          if x.GetIdx() not in bb_list:
            CB_list.append(x)

        if len(CB_list) > 2:
          sys.exit('found more than 2 non-backbone substituents on CA atom')

        ## adjsut for non-zero backbone charges in case of NMETHYL + CABRANCH (fixed_bb_sum correction)
        charge_correct = (sidechain_charge-sc_charge_sum-fixed_bb_sum)/float(len(CB_list))
        for x in CB_list:
          current_charge = x.GetPartialCharge()
          x.SetPartialCharge( current_charge + charge_correct )  
          print('adjusting charge of atom %s by %f: %f -> %f'%(x.GetName(),charge_correct,current_charge,x.GetPartialCharge()))

      ### get final partial charge sum for side chain
      adjusted_sc_charge_sum = 0
      for atom in mol.GetAtoms():
        if atom.GetIdx() not in bb_list:
          adjusted_sc_charge_sum += atom.GetPartialCharge()
      print('adjusted side chain charge = %f\n'%(adjusted_sc_charge_sum))

      ### get final partial charge sum for full residue
      total_sum = 0
      for atom in mol.GetAtoms():
        if (atom.GetIdx() not in ignore_list) and (atom.GetIdx() != upper.GetIdx()) and (atom.GetIdx() != lower.GetIdx()):
          total_sum += atom.GetPartialCharge()
      print('final sum of partial charges = %f\n'%(total_sum))


      for bond in mol.GetBonds():
        if bond.GetBgn().GetType() == 'H' or bond.GetEnd().GetType() == 'H':
          continue
        print('getting TorRule for bond between %s and %s'%(bond.GetBgn().GetType(),bond.GetEnd().GetType()))
        torrule = torLib.GetTorRule(mol, bond)
        if torrule:
          print('torsion between %5s and %5s:'%(bond.GetBgn().GetType(), bond.GetEnd().GetType()), end='' )
          torsions = OEDoubleVector()
          OEGetTorValues(torrule, torsions)
          tors = ['%i'%(degrees(x)) for x in torsions]
          print('  %4i -- %4i    '%(int(bond.GetBgn().GetIdx())+1,int(bond.GetEnd().GetIdx())+1), end='')
          for tor in tors:
            print('%4s '%(tor), end='')
          print('')
        else:
          continue

      ### write only a single conformation to mol2 file
      ### more than one conf will cause molfile_to_params_polymer.py to fail
      print('\nwriting output mol2')
      for conf in mol.GetConfs():
        OEWriteMolecule(ofs, conf)
        break
      ofs.close()

      ### append footer to mol2 file which is used by molfile_to_params_polymer.py
      fout = open(outfile,'a')
      fout.write('@ROSETTA\n')
      fout.write('M  ROOT %i\n'%(N.GetIdx()+1))
      fout.write('M  POLY_N_BB %i\n'%(N.GetIdx()+1))
      fout.write('M  POLY_CA_BB %i\n'%(CA.GetIdx()+1))
      fout.write('M  POLY_C_BB %i\n'%(C.GetIdx()+1))
      fout.write('M  POLY_O_BB %i\n'%(O.GetIdx()+1))
      if NMETH:
        fout.write('M  POLY_CM_BB %i\n'%(CM.GetIdx()+1))
      fout.write('M  POLY_IGNORE')
      for x in ignore_list:
        fout.write(' %i'%(x+1))
      fout.write('\nM  POLY_LOWER %i\n'%(lower.GetIdx()+1))
      fout.write('M  POLY_UPPER %i\n'%(upper.GetIdx()+1))
      fout.write('M  POLY_CHG 0\n')
      fout.write('M  POLY_PROPERTIES')
      if PEPTOID:
        fout.write(' PEPTOID ACHIRAL_SIDECHAIN')
      else:
        fout.write(' PROTEIN ALPHA_AA L_AA')
      if NMETH:
        fout.write(' N_METHYLATED')
      if NCYCLE:
        fout.write(' NCYCLE')
      if CABRANCH or CACYCLE:
        fout.write(' CABRANCH')
      fout.write('\n')
      fout.write('M  END\n')

  return 0

if __name__ == "__main__": 
  sys.exit(main(sys.argv))

