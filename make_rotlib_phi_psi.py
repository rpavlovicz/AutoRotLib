#!/usr/bin/env python3

##!/Users/rpavlovicz/anaconda3/envs/oepython3/bin/python
##!/usr/bin/env python

import os,sys
import fileinput
import itertools
import argparse
from operator import itemgetter, attrgetter
from collections import defaultdict
from numpy import average, std, linspace, sin, cos, deg2rad, rad2deg, arctan2, sqrt, e
from numpy import pi as PI
from openeye.oechem import *
from openeye.oeomega import *
from openeye.oequacpac import *
from openeye.oeszybki import *

### ALL ANGLES ARE KEPT IN DEGREES FROM 0 TO 360

### MakeRotLibMover.cc uses kbT = 1.4 which they found to reproduce reasonable rotamer probabilities
### for 'noncanonical' LEU compared to Dunbrack LEU

# global
#kbt = 0.616033 ## for T = 37 C = 310 K
kbt = 4.0
# global

class ChiDef:
  """ stores information regarding rotamer CHI definitions """
  """ including atoms names from the mol_file_to_params_polymer output """
  """ as well as symmetry information and ideal torision values """
  def __init__(self, atoms):
    self.atom_names = atoms     # list of chi values
    self.symmetry = None        # symmetry number
    self.ideal_torsions = None  # list of expected values
    self.semi_rotameric = None  # real value of starting value if rotamer is determined semi_rotameric
    self.proton_chi = None      # currently not used?

  def set_symmetry(self, symm_no):
    self.symmetry = symm_no

  def set_ideal_torsions(self, torsions):
    self.ideal_torsions = torsions

  def set_semi_rotameric(self, sr):
    self.semi_rotameric = sr


class SemiRot:
  """ class to aid in the claculation of semirotameric probabilities/distributions """
  def __init__(self, chis, probability, stds):
    self.chis = chis      # list of chi values
    self.prob = probability    # probability of rotamer
    self.stds = stds      # list of standard deviations
    self.prob_dist = None    # list of probabilities over range
    self.bins = None      # list of rotameric CHI bins

  def set_chis(self, new_chis):
    self.chis = new_chis

  def append_chi(self, new_chi):
    self.chis.append(new_chi)

  def append_std(self, new_std):
    self.stds.append(new_std)

  def set_probability(self, prob):
    self.probability = prob

  def set_prob_dist(self, prob_distribution):
    self.prob_dist = prob_distribution

  def set_std_devs(self, std_dev_list):
    self.stds = std_dev_list

  def set_bins(self, bin_list):
    self.bins = bin_list


class Rotamer:
  """ stores rotamer chi values and minimized energy """
  ### TODO? make sure chis is a list and all entries are of the same length ###
  def __init__(self, chis, energy):
    self.chis = chis        # list of chi values
    self.energy = energy      # minimized energy of computed rotamer
    self.cluster_id = None      # id of closest centroid
    self.centroid_distances = None  # the distance to each centroid
    self.ncentroids = None      # only assigned to Rotamers[0] 
    self.boltzmann_weight = None  # boltzmann-weighted probability based on current cluster
    self.bins = None        # list of 4 bin values, originally only assoicated with centroids
    self.stds = None        # standard deviations for each chi value
    self.conf = None        # energy-minimzied OEConfBase object that defines this Rotamer object
    self.initial_chis = None    # list of initial CHI values used to get chis after Szybki minimization

  def set_chis(self, chi_list):
    self.chis = chi_list

  def set_cent_dist(self, cent_distances):
    self.centroid_distances = cent_distances

  def set_cluster_id(self, clust_id):
    self.cluster_id = clust_id

  def set_ncentroids(self, ncent):
    self.ncentroids = ncent

  def set_boltzmann_weight(self, boltz):
    self.boltzmann_weight = boltz

  def set_bins(self, bin_list):
    self.bins = bin_list

  def set_stds(self, std_list):
    self.stds = std_list

  def set_conf(self, min_conf):
    self.conf = min_conf

  def set_initial_chis(self, initial_chi_list):
    self.initial_chis = initial_chi_list


def pad_list_with_zeros( inlist, final_length ):
  while len(inlist) < final_length:
    inlist.append(0)
  return inlist


def circ_mean( angles ):
  """ compute circular mean of input list of angles (in degrees) """

  x = 0; y = 0
  for angle in angles:
    x += sin(deg2rad(angle))
    y += cos(deg2rad(angle))
  final_ave = rad2deg(arctan2(x,y))
  ### keep return value in range of 0 to 360
  if final_ave < 0:
    return final_ave+360.0
  else:
    return final_ave


def weighted_circ_mean( rotamers, chi_index ):
  """ compute weighted circular mean for given list of rotamers and chi_index """
  """ rotamers[x].chis is expected to be in degrees """

  x = 0; y = 0
  for rotamer in rotamers:
    x += sin(deg2rad(rotamer.chis[chi_index]))*rotamer.boltzmann_weight
    y += cos(deg2rad(rotamer.chis[chi_index]))*rotamer.boltzmann_weight
  final_ave = rad2deg(arctan2(x,y))
  ### keep return value in range of 0 to 360
  if final_ave < 0:
    return final_ave+360.0
  else:
    return final_ave


def circ_mean_symm( angles, symm_no ):
  """ compute circular mean of input list of angles (in degrees) """

  ### compute two means
  ### 1.) normal circular mean
  ### 2.) circular mean for data wrapped from 0->180 to -90->90 or -60->60
  ### return the value for the mean with the smallest average distance of all data to centroid

  ### 1.) normal circular mean
  x = 0; y = 0
  for angle in angles:
    x += sin(deg2rad(angle))
    y += cos(deg2rad(angle))
  final_ave = rad2deg(arctan2(x,y))
  if final_ave < 0:
    final_ave_1 = final_ave+360.0
  else:
    final_ave_1 = final_ave
  dist_1 = [ angle_dist( angle, final_ave_1, symm_no ) for angle in angles ]

  ### 2.) circular mean for data wrapped from -90->90 or -60->60
  new_angles = [wrap_angle(angle, -180/float(symm_no), 180/float(symm_no), symm_no) for angle in angles]
  x = 0; y = 0
  for angle in new_angles:
    x += sin(deg2rad(angle))
    y += cos(deg2rad(angle))
  final_ave = rad2deg(arctan2(x,y))
  if final_ave < 0:
    final_ave_2 = final_ave+360.0/float(symm_no)
  else:
    final_ave_2 = final_ave
  dist_2 = [ angle_dist( angle, final_ave_2, symm_no ) for angle in angles ]

  if average(dist_1) < average(dist_2):
    return final_ave_1
  else:
    return final_ave_2


def weighted_circ_mean_symm( rotamers, chi_index, symm_no ):
  """ compute weighted circular mean for given list of rotamers and chi_index """

  ## TODO: try to merge weighted and normal circ_mean_symm
  ## TODO: also try to merge with non-symm version

  ### compute two means
  ### 1.) normal circular mean
  ### 2.) circular mean for data wrapped from 0->180 to -90->90 or -60->60
  ### return the value for the mean with the smallest average distance of all data to centroid

  angles = [ rotamer.chis[chi_index] for rotamer in rotamers ]
  ### 1.) normal circular mean
  x = 0; y = 0
  for rotamer in rotamers:
    x += sin(deg2rad(rotamer.chis[chi_index]))*rotamer.boltzmann_weight
    y += cos(deg2rad(rotamer.chis[chi_index]))*rotamer.boltzmann_weight
  final_ave = rad2deg(arctan2(x,y))
  if final_ave < 0:
    final_ave_1 = final_ave+360.0
  else:
    final_ave_1 = final_ave
  dist_1 = [ angle_dist( angle, final_ave_1, symm_no ) for angle in angles ]

  ### 2.) circular mean for data wrapped from -90->90 if symm_no == 2
  ###     or wrapped from -60->60 if symm_no == 3
  new_angles = [wrap_angle(angle, -180/float(symm_no), 180/float(symm_no), symm_no) for angle in angles]

  x = 0; y = 0
  for rotamer in rotamers:
    wrapped_angle = wrap_angle(rotamer.chis[chi_index], -180/float(symm_no), 180/float(symm_no), symm_no)
    x += sin(deg2rad( wrapped_angle ))*rotamer.boltzmann_weight
    y += cos(deg2rad( wrapped_angle ))*rotamer.boltzmann_weight
  final_ave = rad2deg(arctan2(x,y))
  if final_ave < 0:
    final_ave_2 = final_ave+360.0/float(symm_no)
  else:
    final_ave_2 = final_ave
  dist_2 = [ angle_dist( angle, final_ave_2, symm_no ) for angle in angles ]

  if average(dist_1) < average(dist_2):
    return final_ave_1
  else:
    return final_ave_2


def wrap_angle(angle_in, min_val, max_val, symm_no):
  """ return angle in range of [min_val,max_val) """
  if max_val < min_val:
    sys.exit('error: angle range of %f to %f is not valid'%(min_val,max_val))

  new_angle = angle_in
  while new_angle < min_val:
    new_angle += 360/float(symm_no)

  while new_angle >= max_val:
    new_angle -= 360/float(symm_no)

  return new_angle


def periodic_range( angle_in ):
  """ return angle in range of [-180,180) """
  if ( angle_in >= 180.0 ):
    return angle_in-360
  elif ( angle_in < -180 ):
    return angle_in+360
  else:
    return angle_in


def check_angle_range( angle_in, boundary ):
  """ make sure angle is within range of 0.0 to boundary """
  """ normally boundary is 360, but for symmetric cases, """
  """ boundary may be 180 or 120 if C2 or C3 symmetry """

  if ( angle_in < 0.0 ) or ( angle_in > boundary ):
    return False
  else:
    return True


def angle_dist( angle1, angle2, symm_no = 1 ):
  """ calculate dis  return atan( x/y )tance between two angles """
  """ angles should be between 0 and 360 """
  wrap_boundary = 360/symm_no

  if not check_angle_range( angle1, wrap_boundary ):
    sys.exit('ERROR: angle_dist requires angles to be within 0.0 and %i.0. input angle = %6.1f'%(wrap_boundary,angle1))
  elif not check_angle_range( angle2, wrap_boundary ):
    sys.exit('ERROR: angle_dist requires angles to be within 0.0 and %i.0. input angle = %6.1f'%(wrap_boundary,angle2))

  d1 = abs(angle1-angle2)
  d2 = wrap_boundary-d1
  return d1 if d1<d2 else d2


def calc_rot_dist( rot1, rot2, chi_defs ):
  """ calculate distance between two rotamers """
  """ this is the squared mean distance between chi values """
  """ use chi_defs only for symmetry information """

  ### make sure number of chis is the same beween rot1 and rot2
  if len(rot1.chis) != len(rot2.chis):
    sys.exit('ERROR: unequal number of CHI values between %s and %s'%(rot1.chis,rot2.chis))

  ### also assure number of chi_defs is the same as the number of chis in each rotamer
  ### although this doesn't necessarily have to be true in the case of a proton chi that is not parameterized? 
  if len(rot1.chis) != len(chi_defs):
    sys.exit('ERROR: number of torsions in rotamer != number of CHI definitions')

  square_distance = 0
  for x in range(len(rot1.chis)):
    if chi_defs[x].symmetry == 2:
      ### in case of symmetric torsion, compute two angle distances
      ### d1 = normal distance with wrapping around 0/360 boundary
      ### d2 = distance with wrapping around 0/180 C2 boundary
      d1 = angle_dist(rot1.chis[x], rot2.chis[x])**2
      d2 = angle_dist(rot1.chis[x], rot2.chis[x], 2)**2
      square_distance += min(d1,d2)
    elif chi_defs[x].symmetry == 3:
      d1 = angle_dist(rot1.chis[x], rot2.chis[x])**2
      d2 = angle_dist(rot1.chis[x], rot2.chis[x], 3)**2
      square_distance += min(d1,d2)
    else:
      square_distance += angle_dist(rot1.chis[x], rot2.chis[x])**2

  return sqrt( square_distance/float(len(rot1.chis)) )


def calc_all_dist( rotamers, centroids, chi_defs ):   
  """ calculate distance between all rotamers and all centroid positions """
  """ store this info in the Rotamers class for later centroid assignment """
  """ chi_defs used only for symmetry information """

  ### set number of centroids in rotamers
  if rotamers[0].ncentroids == None:
    rotamers[0].set_ncentroids( len(centroids) )
  else:
    if rotamers[0].ncentroids != len(centroids):
      sys.exit('ERROR: attempting to recalculate centroid distances for number of cluster centers differing from previous calculation') 

  for rotamer in rotamers:
    distances = []
    for centroid in centroids:
      distances.append( calc_rot_dist( rotamer, centroid, chi_defs ) )
    rotamer.set_cent_dist(distances)


def assign_cluster_number( rotamers ):
  """ assign a cluster for each rotamer in rotamers """
  """ based on closest distance to a centroid position calculated in calc_all_dist """

  for rotamer in rotamers:
    rotamer.set_cluster_id( min(enumerate(rotamer.centroid_distances), key=itemgetter(1))[0] )


def calc_boltzmann_weights( rotamers ):
  """ calculate boltzmann-weighted probability for each rotamer """
  """ where partition function is calculated for each cluster """

  for x in range(rotamers[0].ncentroids):
    cluster_rotamers = [rotamer for rotamer in rotamers if rotamer.cluster_id == x]
    if len(cluster_rotamers) == 0: continue  # this is usually due to low sampling interval or high number of CHI bins
    min_energy = min([rotamer.energy for rotamer in cluster_rotamers])
    print('cluster %i has %i members with min_energy = %f'%(x,len(cluster_rotamers),min_energy))
    clust_Z = sum([e**(-float(rotamer.energy-min_energy)/float(kbt)) for rotamer in cluster_rotamers])
    for rotamer in cluster_rotamers:
      rotamer.set_boltzmann_weight( e**(-float(rotamer.energy-min_energy)/float(kbt))/clust_Z )


def calc_cluster_centroids( rotamers, chi_defs ):
  """ recenter centroids based on previous assignment """
  """ computer probability-weighted average if boltzmann_weight field is filled """

  if rotamers[0].cluster_id == None:
    sys.exit('ERROR: rotamers have not yet been assigned centroid values')

  ### TODO? add check if rotamers[0].ncentroids == len(product(*[chi.ideal_torsions for chi in chi_defs]))?
  ### the use of ncentroids in first member of rotamers is very ugly 
  ### TODO: there can also be problems in that len(rotamers[0].chis) may not == len(chi_defs)

  nchi = len(rotamers[0].chis); new_centroids = []
  ### iterate over centroids numbers and compute avereage chi angles for all cluster current members
  for x in range(rotamers[0].ncentroids):
    clust_rotamers = [rotamer for rotamer in rotamers if rotamer.cluster_id == x]
    new_chis = []
    for chi_index in range(nchi):
      ### check if CHI is symmetric
      if (chi_defs[chi_index].symmetry == 2) or (chi_defs[chi_index].symmetry == 3):
        if rotamers[0].boltzmann_weight != None:
          new_chis.append( weighted_circ_mean_symm( clust_rotamers, chi_index, chi_defs[chi_index].symmetry ) )
        else:
          new_chis.append( circ_mean_symm( [rotamer.chis[chi_index] for rotamer in clust_rotamers], chi_defs[chi_index].symmetry ) )
      else:
        ### arithmetic mean is not appropriate here due to circular quantities!
        ### if rotamer.boltzman_weights present, compute weighted circular average
        if rotamers[0].boltzmann_weight != None:
          new_chis.append( weighted_circ_mean( clust_rotamers, chi_index ) )
        ### else compute un-weighted circular average
        else:
          new_chis.append( circ_mean( [rotamer.chis[chi_index] for rotamer in clust_rotamers] ) )
    ### append new centroid rotamers with empty energy fields
    new_centroids.append(Rotamer(new_chis,None))  

  return new_centroids


def angle_wrap( chi_list ):
  """ convert 0-360 chi lists to -180-180 chi lists """

  wrapped_data = []
  for chi in chi_list:
    if chi > 180.0:
      wrapped_data.append(chi-360.0)
    else:
      wrapped_data.append(chi)

  return wrapped_data


def get_bin_assignments(chis,torsions):
  """ create bin assignments for initial centroids """

  if len(chis) > 4:
    sys.exit('ERROR: Rosetta cannot handle side chains with more than 4 torsions')

  bins = []
  for x in range(len(chis)):
    bins.append(torsions[x].index(chis[x])+1)
  while len(bins) < 4:
    bins.append(0)

  return bins


def get_torlib(rotlib_file):
  """ read torsion library info generated in OE stderr """
  """ this includes info on symmetric bonds """

  with open(rotlib_file) as f:
    torlib = f.readlines()

  symmetric_bonds = []

  tors = defaultdict(list)
  for line in fileinput.input(rotlib_file):
    if line.startswith('torsion between'):
      ls = line.rsplit()
      key = '%s,%s'%(ls[5],ls[7])
      tor_list = [float(x) for x in ls[8:]]
      tors[key] = tor_list
    if line.startswith('Symmetric'):
      ls = line.rsplit()
      symmetric_bonds.append('%s,%s'%(ls[3],ls[4]))
      symmetric_bonds.append('%s,%s'%(ls[4],ls[3]))

  return tors, symmetric_bonds


def get_chis( conf, chi_defs, atom_map, symm ):
  """ get CHI values from OEConf using atom names defined in chis """
  """ return values in degrees between 0 and 360 """

  chi_vals = []
  for chi in chi_defs:
    a = conf.GetAtom( OEHasAtomName(atom_map[ chi.atom_names[0] ]) )
    b = conf.GetAtom( OEHasAtomName(atom_map[ chi.atom_names[1] ]) )
    c = conf.GetAtom( OEHasAtomName(atom_map[ chi.atom_names[2] ]) )
    d = conf.GetAtom( OEHasAtomName(atom_map[ chi.atom_names[3] ]) )
    chi_val = OEGetTorsion(conf, a, b, c, d)*180.0/PI

    ### bring initial chi_val into [0,360)
    chi_val = chi_val if chi_val >= 0.0 else chi_val+360.0
#    print('chi for %s %s %s %s = %f'%(a.GetName(),b.GetName(),c.GetName(),d.GetName(),chi_val))

    ### perform symmetry correction here
    if symm and (chi.symmetry == 2 or chi.symmetry == 3):
      symm_corr_chi = chi_val
      while symm_corr_chi > 360/float(chi.symmetry):
        symm_corr_chi -= 360/float(chi.symmetry)
      chi_vals.append(symm_corr_chi)  
    else:
      chi_vals.append(chi_val)    

  chi_vals.append(conf.GetEnergy())

  return chi_vals


def get_chi_atoms( conf, chi, atom_map ):
  """ read atom_map files and return OEAtom object for given chi """

  a1 = conf.GetAtom( OEHasAtomName(atom_map[ chi[0] ]) )
  a2 = conf.GetAtom( OEHasAtomName(atom_map[ chi[1] ]) )
  a3 = conf.GetAtom( OEHasAtomName(atom_map[ chi[2] ]) )
  a4 = conf.GetAtom( OEHasAtomName(atom_map[ chi[3] ]) )

  return a1, a2, a3, a4


def detect_symmetry( conf, ba1, ba2 ):
  """ atoms ba1 and ba2 were determined to be a symmetric bond """
  """ ba2 should be the atom that is furthest from the backbone """
  """ check how many substituents are on ba2 that are not ba1 """
  """ make sure they are the same, then report the number """
  """ this should be the symmetry number """
  sub = [];
  for bond in ba2.GetBonds():
    if bond.GetBgn() == ba1 or bond.GetEnd() == ba1: continue
    elif bond.GetBgn() == ba2:
      sub.append(bond.GetEnd().GetType())
    else:
      sub.append(bond.GetBgn().GetType())
  print('atom types bound to %s = '%(ba2.GetName()),sub)
  if len(sub) == sub.count(sub[0]):
    print('   identified bond with %i-fold symmetry'%(len(sub)))
    return len(sub)
  elif (ba2.GetType() == "N.pl3") and ("O.3" in sub) and ("O.2" in sub) and (len(sub) == 2):
    print('   identified nitro group with 2-fold symmetry')
    return 2
  else:
    sys.exit('found more than one type of substituent extending from "symmetric" bond')


def get_final_rotamers( rotamers, centroids ):
  """ from Rotamer set, and initial centroids (for # and bin info of final rotamers) """
  """ get lowest energy conformer per cluster and calculate probabilities """
  final_rotamers = []

  Z = 0  ### partition function for probability calculation
  for x in range(len(centroids)):
    ### get conformer info for those clustered into cluster_id x
    energies = [ [rotamer.energy, rotamer.centroid_distances[x], rotamer.chis, rotamer.conf] for rotamer in rotamers if rotamer.cluster_id == x]

    ### check if a cluster_id has no members, if so, create a dummy rotamer for this id with ideal CHI angles and zero probability
    if len(energies) == 0:
      ### make this a warning -- assign chi bin cluster to ideal values with 0 probability
      dummy_rot = Rotamer(list(centroids[x].chis),1e6)
      dummy_rot.set_bins( centroids[x].bins )
      dummy_rot.set_boltzmann_weight( 0.0 )
      final_rotamers.append(dummy_rot)
      print('WARNING: found cluster with no members')      
    else: 
      ### sort by energies, with distance from centroid as second sorting criteria
      energies.sort()
      rot_chis = energies[0][2]; rot_energy = energies[0][0]
      final_rotamers.append(Rotamer(rot_chis,rot_energy))
      final_rotamers[-1].set_conf(energies[0][3])
      final_rotamers[-1].set_bins( centroids[x].bins )
      Z += e**(-float(rot_energy)/float(kbt))

  min_energy = min([rotamer.energy for rotamer in final_rotamers])
  Z = sum([e**(-float(rotamer.energy-min_energy)/float(kbt)) for rotamer in final_rotamers])

  for rotamer in final_rotamers:
    rotamer.set_boltzmann_weight( e**(-float(rotamer.energy-min_energy)/float(kbt)) / Z )

  return final_rotamers


def get_std_devs( mol, chi_vals, chi_defs, cut_bonds, cut_bonds_2, atom_map ):
  """ compute std_dev for all chis """

  ### create single point SZYBKI object for rescoring conformers
  szybOpts_sp = OESzybkiOptions()
  szybOpts_sp.SetRunType(OERunType_SinglePoint)
  szybOpts_sp.GetGeneralOptions().SetForceFieldType(OEForceFieldType_MMFF94S) # MMFF94S, MMFF94, MMFF_AMBER, MMFFS_AMBER
  szybOpts_sp.GetSolventOptions().SetSolventModel(OESolventModel_Sheffield) # Sheffield, NoSolv
  szybOpts_sp.GetSolventOptions().SetSolventDielectric(3.0)
  szybOpts_sp.GetSolventOptions().SetChargeEngine(OEChargeEngineNoOp())
  sz_sp = OESzybki(szybOpts_sp)
  sz_results_sp = OESzybkiResults()

  ### compute std_dev for all chis of an input conformation
  std_devs = []
  for x in range(len(chi_vals)):
    ### for each CHI, start with the original conformation
    ### requires creating copy of original mol, otherwise maintains final state of CHI sweep
    mol_copy = mol.CreateCopy()
    conf = mol_copy.GetActive()

    ### TODO: check that phi/psi are set correctly!!!

    ### get atoms that are part of the CHI torsion to be sampled
    ### read in from ChiDefs (Rosetta naming) using atom_map to convert to OE naming
    a = conf.GetAtom( OEHasAtomName(atom_map[ chi_defs[x].atom_names[0] ]) )
    b = conf.GetAtom( OEHasAtomName(atom_map[ chi_defs[x].atom_names[1] ]) )
    c = conf.GetAtom( OEHasAtomName(atom_map[ chi_defs[x].atom_names[2] ]) )
    d = conf.GetAtom( OEHasAtomName(atom_map[ chi_defs[x].atom_names[3] ]) )
    initial_chi = conf.GetTorsion(a,b,c,d)*180/PI
    initial_energy = conf.GetEnergy()
    print('*** initial CHI%i = %6.2f; score = %f'%(x+1,initial_chi,conf.GetEnergy()))

    ### move CHI by +/- 0.1 degree intervals until
    ### energy difference >= 0.5 or degree change >= 30.0
    ### store final degree change and std_dev for CHI 
    bin_energy_tolerance = 0.5; chi_delta = 0.1
    energy_change = 0; y = 0
    while (energy_change <= bin_energy_tolerance and y*chi_delta <= 30.0):
      y += 1
      # + delta degrees
      new_chi_plus = initial_chi + chi_delta*y

      if (len(cut_bonds_2) > 0) and (chi_defs[x].atom_names[3] == 'N' or chi_defs[x].atom_names[3] == 'NV'):
        delete_cut_bonds( conf, cut_bonds_2, atom_map )
      elif len(cut_bonds) > 0:
        delete_cut_bonds( conf, cut_bonds, atom_map )

      conf.SetTorsion(a,b,c,d,new_chi_plus*PI/180.0)

      if (len(cut_bonds_2) > 0) and (chi_defs[x].atom_names[3] == 'N' or chi_defs[x].atom_names[3] == 'NV'):
        rebuild_cut_bonds( conf, cut_bonds_2, atom_map )
      elif len(cut_bonds) > 0:
        rebuild_cut_bonds( conf, cut_bonds, atom_map )

      sz_sp(conf,sz_results_sp)
      energy_change_plus = abs(initial_energy-conf.GetEnergy())


      # - delta degrees
      new_chi_minus = initial_chi - chi_delta*y

      if (len(cut_bonds_2) > 0) and (chi_defs[x].atom_names[3] == 'N' or chi_defs[x].atom_names[3] == 'NV'):
        delete_cut_bonds( conf, cut_bonds_2, atom_map )
      elif len(cut_bonds) > 0:
        delete_cut_bonds( conf, cut_bonds, atom_map )

      conf.SetTorsion(a,b,c,d,new_chi_minus*PI/180.0)

      if (len(cut_bonds_2) > 0) and (chi_defs[x].atom_names[3] == 'N' or chi_defs[x].atom_names[3] == 'NV'):
        rebuild_cut_bonds( conf, cut_bonds_2, atom_map )
      elif len(cut_bonds) > 0:
        rebuild_cut_bonds( conf, cut_bonds, atom_map )


      energy_change_minus = abs(initial_energy-conf.GetEnergy())
      energy_change = max([energy_change_plus,energy_change_minus])
      print('step %3i: delta = %4.1f; max energy change = %7.4f'%(y,chi_delta*y,energy_change))
    print('final energy_change = %4.2f at %4.2f degree change'%(energy_change, chi_delta*y))
    std_devs.append(chi_delta*y)
    conf.Delete()

  while len(std_devs) < 4:
    std_devs.append(0)

  return std_devs


def get_semirotameric( mol, chis, chi_defs, atom_map, semirange ):

  ### create single point SZYBKI object for rescoring conformers
  szybOpts_sp = OESzybkiOptions()
  szybOpts_sp.SetRunType(OERunType_SinglePoint)
  szybOpts_sp.GetGeneralOptions().SetForceFieldType(OEForceFieldType_MMFF94S) # MMFF94S, MMFF94, MMFF_AMBER, MMFFS_AMBER
  szybOpts_sp.GetSolventOptions().SetSolventModel(OESolventModel_Sheffield) # Sheffield, NoSolv
  szybOpts_sp.GetSolventOptions().SetSolventDielectric(3.0)
  szybOpts_sp.GetSolventOptions().SetChargeEngine(OEChargeEngineNoOp())
  sz_sp = OESzybki(szybOpts_sp)
  sz_results_sp = OESzybkiResults()

  ### create working copy of molecule
  mol_copy = mol.CreateCopy()
  conf = mol_copy.GetActive()

  ### set rotameric chi values
  for x in range(len(chis)):
    a1 = conf.GetAtom( OEHasAtomName(atom_map[ chi_defs[x].atom_names[0] ]) )
    a2 = conf.GetAtom( OEHasAtomName(atom_map[ chi_defs[x].atom_names[1] ]) )
    a3 = conf.GetAtom( OEHasAtomName(atom_map[ chi_defs[x].atom_names[2] ]) )
    a4 = conf.GetAtom( OEHasAtomName(atom_map[ chi_defs[x].atom_names[3] ]) )
    initial_chi = conf.GetTorsion(a1,a2,a3,a4)*180/PI
    conf.SetTorsion(a1,a2,a3,a4,chis[x]*PI/180.0)

  ### scan semirotmeric range
  a1 = conf.GetAtom( OEHasAtomName(atom_map[ chi_defs[-1].atom_names[0] ]) )
  a2 = conf.GetAtom( OEHasAtomName(atom_map[ chi_defs[-1].atom_names[1] ]) )
  a3 = conf.GetAtom( OEHasAtomName(atom_map[ chi_defs[-1].atom_names[2] ]) )
  a4 = conf.GetAtom( OEHasAtomName(atom_map[ chi_defs[-1].atom_names[3] ]) )
  energy_distribution = []
  for sr_angle in semirange:
    conf.SetTorsion(a1,a2,a3,a4,sr_angle*PI/180.0)
    sz_sp(conf,sz_results_sp)
    energy_distribution.append(conf.GetEnergy())

  return energy_distribution


def boltzmann_dist( energy_list ):
  """ calculate boltzmann weighted probabilities given a list of energies """

  min_energy = min(energy_list)
  Z = sum([ e**(-float(energy-min_energy)/float(kbt)) for energy in energy_list ])

  return [ e**(-float(energy-min_energy)/float(kbt))/Z for energy in energy_list ]


def delete_cut_bonds( conf, cut_bonds, atom_map ):
  ### check if bond is in cut_bonds. if so, delete the bond
  for bond in conf.GetBonds():
    for i,cb in enumerate(cut_bonds):
      if bond.GetBgn().GetName() == atom_map[ cb[0] ] and bond.GetEnd().GetName() == atom_map[ cb[1] ]:
        conf.DeleteBond(bond)
        #print('deleting bond between %s and %s'%(bond.GetBgn().GetName(), bond.GetEnd().GetName()))
        OEFindRingAtomsAndBonds(conf)
        # add bond order to cb in cut_bonds for when we rebuild later
        cut_bonds[i].append(bond.GetOrder())
        continue
      if bond.GetBgn().GetName() == atom_map[ cb[1] ] and bond.GetEnd().GetName() == atom_map[ cb[0] ]:
        conf.DeleteBond(bond)
        #print('deleting bond between %s and %s'%(bond.GetBgn().GetName(), bond.GetEnd().GetName()))
        OEFindRingAtomsAndBonds(conf)
        # add bond order to cb in cut_bonds for when we rebuild later
        cut_bonds[i].append(bond.GetOrder())
        continue


def rebuild_cut_bonds( conf, cut_bonds, atom_map ):
  ### rebuild any cut bonds
  for cb in cut_bonds:
    a1 = conf.GetAtom( OEHasAtomName(atom_map[ cb[0] ]) )
    a2 = conf.GetAtom( OEHasAtomName(atom_map[ cb[1] ]) )
    conf.NewBond(a1,a2,cb[2])  # cb[2] should be bond order set in delete_cut_bonds


def read_properties(infile):
  """ read footer of mol2 files made by charge_and_correct script """
  PEPTOID = NMETH = NCYCLE = False

  for line in fileinput.input(infile):
    if line.startswith('M  POLY_PROPERTIES'):
      ls = line.rsplit()
      fileinput.close()
  if 'PEPTOID' in ls:
    PEPTOID = True
  if 'NMETH' in ls:
    NMETH = True
  if 'NCYCLE' in ls:
    NCYCLE = True

  return(PEPTOID,NMETH,NCYCLE)

def check_arg(args=None):
  parser = argparse.ArgumentParser(description="create N input files for MakeRotLib, where N = n_phi_bins*n_psi_bins")
  parser.add_argument('-n', '--aa_name', required=True, type=str, help='three letter name for residue')
  parser.add_argument('--phi', required=True, type=int, help='phi angle')
  parser.add_argument('--psi', required=True, type=int, help='psi angle')
  parser.add_argument('--omega', required=False, type=int, default=180, help='omega angle')
  parser.add_argument('--allow_semirotameric', required=False, type=bool, default=False, help='allow autodetection and generation of semirotameric residue libraries')
  parser.add_argument('--test', required=False, type=bool, default=False, help='set 45 degree increments for all chis, print out mol2 of final rotamers')
  results = parser.parse_args(args)
  return (results.aa_name, results.phi, results.psi, results.omega, results.allow_semirotameric, results.test)


def main(argv=[__name__]):

  aa_name, phi, psi, omega, allow_semirotameric, TEST_RUN = check_arg(sys.argv[1:])

  ### make sure input aa_name is three characters long
  if len(aa_name) != 3:
      sys.exit('aa_name should be three characters long')

  pwd = os.getcwd()

  infile = '%s/%s_AM1BCC.mol2'%(pwd,aa_name)
  ifs = oemolistream()
  if not ifs.open(infile):
    OEThrow.Fatal("Unable to open %s for reading" % infile)
  workdir = os.path.dirname(infile)
  print('workdir = %s'%(workdir))
  workdir = '.'

  PEPTOID, NMETH, NCYCLE = read_properties(infile)
  print('PEPTOPID = %s'%(PEPTOID))
  print('NMETH = %s'%(NMETH))
  print('NCYCLE = %s'%(NCYCLE))
  print('TEST = %s'%(TEST_RUN))

  torlib_log = '%s_torlib_OE'%(aa_name)
  ### make sure torsion library exists
  if not os.path.isfile(torlib_log):
    sys.exit('could not find torsion library file: %s'%(torlib_log))


  ### set up Szybki for side chain minimization with fixed backbone
  ### dependent on single CC(=O)NCC(=O)NC match
  szybOpts = OESzybkiOptions()
  szybOpts.GetOptOptions().SetOptimizerType(OEOptType_SD) # options = OEOptType_NEWTON, OEOptType_BFGS, OEOptType_CG, OEOptType_SD, OEOptType_SD_BFGS, SD_CG
  szybOpts.GetOptOptions().SetGradTolerance(0.001) # default = 0.1 except for Newton-Raphson = 0.00001 
  szybOpts.GetGeneralOptions().SetForceFieldType(OEForceFieldType_MMFF94S)
  szybOpts.GetSolventOptions().SetSolventModel(OESolventModel_Sheffield) # OESolventModel_Sheffield, OESolvent_NoSolv
  szybOpts.GetSolventOptions().SetSolventDielectric(3.0)
  szybOpts.GetSolventOptions().SetChargeEngine(OEChargeEngineNoOp())
  sz = OESzybki(szybOpts)
  sz.FixAtoms("CC(=O)NCC(=O)NC") # N-to-C
  sz_results = OESzybkiResults()

  ### map Rosetta params file atom names [ keys ] to OE names [ values ]
  ### atom_map file currently created in molfile_to_params_polymer.py 
  atom_map = {}
  for line in fileinput.input('%s/%s.atom_map'%(workdir, aa_name)):
    ls = line.rsplit()
    atom_map[ ls[1] ] = ls[0]

  ### get OEOmega torsion library
  torlib, symmetric_bonds = get_torlib(torlib_log)

  ### read CHI definitions from params file
  ### ChiDef class constitutes atom_names, ideal torsions, and symmetry
  ### only the names are initially detected upon params file reading
  ### ideal torsions and symmetry are detected from OE torsion library output
  # TODO: detect and define CHI values within OE scripts
  chi_defs = []
  cut_bonds = []
  cut_bonds_2 = []
  virtuals = {}
  PARAMS_BINS = False
  for line in fileinput.input('%s.params'%(aa_name)):
    if line.startswith('NAME '):
      aa_name = line.rsplit()[1]
    if line.startswith('CHI '):
      ls = line.rsplit()
      print(ls,ls[3],ls[4])
      if ls[3] == 'N' and ls[4] == 'CM':
        # skip N-methyl group chi
        continue
      else:
        chi_defs.append(ChiDef([ ls[2], ls[3], ls[4], ls[5] ]))
    if line.startswith('VIRTUAL_SHADOW'):
      virtuals[ line.rsplit()[1] ] = line.rsplit()[2]
    if line.startswith('CUT_BOND'):
      cut_bonds.append([ line.rsplit()[1], line.rsplit()[2] ])
    if line.startswith('NCAA_ROTLIB_NUM_ROTAMER_BINS'):
      PARAMS_BINS = True
  if len(chi_defs) == 0:
    sys.exit('ERROR: no CHI definitions found in params file: %s.params'%(aa_name))

  ### if we have an N-cyclize AA find the opposing 'cut_bond'
  ### actually, this should go for any backbone-cyclized AA
  if NCYCLE and len(cut_bonds) > 0:
    print('verifying cut_bonds:')
    N_cut = False
    for bond in cut_bonds:
      for atom in bond:
        if atom == 'N':
          N_cut = True
    if N_cut == False:
      sys.exit('ERROR: could not identify N-cyclizing bond from params file')

    for chi in chi_defs:
      if chi.atom_names[1] == 'CA':
        cut_bonds_2.append( [chi.atom_names[1], chi.atom_names[2]] )

  ### proton CHI defs
  proton_chi_atom_names = []

  ### get single conf from input OEMol object
  ### identify backbone and torsions, and assign CHI bins
  for mol in ifs.GetOEMols():

    single_conf = mol.GetActive()

    ### assure only one identified instance of ACE/NME-capped backbone
    ss = OESubSearch("CC(=O)NCC(=O)NC")
    OEPrepareSearch(single_conf,ss)
    if not ss.SingleMatch(single_conf):
      sys.exit('ERROR: single backbone match not found in input')

    ### get atoms for setting phi/psi; match results are 0-indexed
    for match in ss.Match(single_conf):
      for ma in match.GetAtoms():
        if   ma.pattern.GetIdx() == 0: bb0 = ma.target  # n-1 CA (for setting omega angle)
        elif ma.pattern.GetIdx() == 1: bb1 = ma.target  # n-1 C
        elif ma.pattern.GetIdx() == 3: bb2 = ma.target  # n   N
        elif ma.pattern.GetIdx() == 4: bb3 = ma.target  # n   CA
        elif ma.pattern.GetIdx() == 5: bb4 = ma.target  # n   C
        elif ma.pattern.GetIdx() == 7: bb5 = ma.target  # n+1 N
        elif ma.pattern.GetIdx() == 8: bb6 = ma.target  # n+1 CA (for setting omega angle)

    for x in range(len(chi_defs)):
      print('chidef %i: '%(x+1),chi_defs[x].atom_names)

    ### match
    torsions = []; chi_atom_names = []; del_list = []
    print('length of chi_defs = %i'%(len(chi_defs)))
    print('\n\nprocessing %s'%(infile))
    for x in range(len(chi_defs)):
      symmetry = None; skip = 0
      print('length of chi_defs = %i'%(len(chi_defs)))
      print('chi_defs[%i] = '%(x),chi_defs[x])
      c1, c2, c3, c4 = get_chi_atoms(single_conf, chi_defs[x].atom_names, atom_map)
      chi_atom_names.append([c1.GetName(), c2.GetName(), c3.GetName(), c4.GetName()])
      print('\nCHI%i: center atoms = %s %s'%(x+1,chi_defs[x].atom_names[1],chi_defs[x].atom_names[2]))
      print('   corresponds to atom names %s %s'%(atom_map[ chi_defs[x].atom_names[1] ], atom_map[ chi_defs[x].atom_names[2] ]))
      idx2 = c2.GetIdx()+1
      idx3 = c3.GetIdx()+1
      print('   corresponds to atom indices %s %s'%(idx2, idx3))
      if ('%s,%s'%(idx2,idx3)) in symmetric_bonds:
        print('   *** bond identified as symmetric by OE ***')
        symm_no = detect_symmetry(single_conf, c2, c3)
        chi_defs[x].set_symmetry(symm_no)

      ### torsions from library can be defined in either direction, check both
      key1 = '%s,%s'%(idx2,idx3); key2 = '%s,%s'%(idx3,idx2)
      foundkey = 0
      if (c1.GetType() == 'H') or (c4.GetType() == 'H'):
        proton_chi_atom_names.append(chi_atom_names[-1])
        print('   *** adding proton rotamer: ',chi_defs[x].atom_names)
        skip = 1
      elif torlib[key1]:
        foundkey = key1
      elif torlib[key2]:
        foundkey = key2
      else:
        ### this should probably be an error?
        ### or prompt user for input?
        ### or use default values?
        ### currently happening when proton torsion is detected in molfile_to_params_polymer
#        if (c1.GetType() == 'H') or (c4.GetType() == 'H'):
#          proton_chi_atom_names.append(chi_atom_names[-1])
#          print('   *** adding proton rotamer: ',chi_defs[x].atom_names)      
#          skip = 1
        if (c2.IsInRing() and c3.IsInRing()):
          foundkey = key1  # arbitrarily select key1 over key2
          torlib[key1] = [-60.0, 60.0, 180.0]
        else:
          sys.exit('ERROR: no torsion values found for CHI%s'%(x+1))  

      if foundkey:
        print('   found %s torsions in %s'%(len(torlib[foundkey]),torlib_log))
        print('  ',torlib[foundkey])
        print('   torsion atom types = %s %s %s %s'%(c1.GetType(),c2.GetType(),c3.GetType(),c4.GetType()))
        ### check for internal guanadinium torsion and remove from chi list if found
        if (c2.GetType() == 'N.pl3' and c3.GetType() == 'C.cat') or (c2.GetType() == 'C.cat' and c3.GetType() == 'N.pl3'):
          print('   *** detected torsion in guanadinium group -- removing from CHI list ***')
          ### TODO: if we get to this point, the .params file needs to be edited to remove this CHI definition
          skip = 1

            ### adjust torsion bin that rotate guanidinium groups 
        if (c2.GetType() == 'C.3' and c3.GetType() == 'N.pl3' and c4.GetType() == 'C.cat') or (c3.GetType() == 'C.3' and c2.GetType() == 'N.pl3' and c1.GetType() == 'C.cat'):
          print('   *** detected torsion rotating guanidinium group -- setting to bin values to -60, 60, 180')
          torlib[foundkey] = [-60.0, 60.0, 180.0]

        ### fix CHI1 if about C.3-C.3 bond -- force -60, 60, 180 bins
        ### this is a weird case, where OE torsion library wants to define 9 torsions about this angle
        ### which it defines very generically:   2   7   6  20 [*:1]~[^3:2]-[^3:3]~[*:4] 
        if chi_defs[x].atom_names[0] == 'N' and c2.GetType() == 'C.3' and c3.GetType() == 'C.3':
          print('   *** found CHI1 about sp3 carbon atoms -- adjusting bin values to -60, 60, 180')
          torlib[foundkey] = [-60.0, 60.0, 180.0]

        ### detect if CHI rotates a ring
        if c2.IsInRing() and c3.IsInRing():
          same_ring = False
          nrings, parts = OEDetermineRingSystems(single_conf)
          if (parts[c2.GetIdx()] == parts[c3.GetIdx()]):
            same_ring = True

          if same_ring:
            ring_size = OEAtomGetSmallestRingSize(c2)
            print('both central torsion atoms are in the same ring')


            ### if this is CHI1, then set the number of ring confo0rmations
            ### and keep all other CHI at 0
            if x == 0:
              if ring_size == 5:
                print('   *** setting torsions for chi %i for ring of size %i -- adjusting bin values to -30 and 30'%(x,ring_size))
                torlib[foundkey] = [-30.0,30.0]
              elif ring_size == 6:
                print('   *** setting torsions for chi %i for ring of size %i -- adjusting bin values to -30 and 30'%(x,ring_size))
                torlib[foundkey] = [-30.0,30.0]
              else:
                sys.exit('not yet configured to handle rings of size %i'%(ring_size))          
            else:
              print('   *** setting torsion for chi %i for ring of size %i -- adjusting bin values to 0'%(x,ring_size))
              torlib[foundkey] = [0.0]


        if ( c3.IsInRing() and c4.IsInRing() )  and not ( c1.IsInRing() and c2.IsInRing() and c3.IsInRing() and c4.IsInRing()):
          print('   * this torsion rotates a ring')
          if (x+1 == len(chi_defs)) and allow_semirotameric:
            print('   *** found terminal CHI rotating a ring; setting to semi-rotameric')
            chi_defs[x].set_semi_rotameric(-30.0)
          elif (x+1 == len(chi_defs)-1) and allow_semirotameric:
            c1_2, c2_2, c3_2, c4_2 = get_chi_atoms(single_conf ,chi_defs[x+1].atom_names, atom_map)
            if (c1_2.GetType() == 'H') or (c4_2.GetType() == 'H'):
              print('   *** found non-terminal CHI rotating a ring with terminal proton chi; setting to semi-rotameric')
              chi_defs[x].set_semi_rotameric(-30.0)

        ### detect if CHI rotates carboxyl group
        if c1.GetType() == 'O.co2':
          print('   *** found O.co2 type')
          nbrs = [atm.GetType() for atm in c2.GetAtoms() if atm not in [c1, c2, c3, c4]]
          if 'O.co2' in nbrs:
            print('   *** idenitified torsion rotating a carboxyl group -- adjusting bin values to -60, -30, 0, 30, 60, 90')
            torlib[foundkey] = [0.0,30.0,60.0,90.0,120.0,150.0]
            if (x+1 == len(chi_defs)) and allow_semirotameric:
              print('   *** found terminal carboxylate -- setting CHI to semi-rotameric')
              chi_defs[x].set_semi_rotameric(-90.0)

        elif c4.GetType() == 'O.co2':
          nbrs = [atm.GetType() for atm in c3.GetAtoms() if atm not in [c1, c2, c3, c4]]
          if 'O.co2' in nbrs:
            print('   *** idenitified torsion rotating a carboxyl group -- adjusting bin values to -60, -30, 0, 30, 60, 90')
            torlib[foundkey] = [0.0,30.0,60.0,90.0,120.0,150.0]
            if (x+1 == len(chi_defs)) and allow_semirotameric:
              print('   *** found terminal carboxylate -- setting CHI to semi-rotameric')
              chi_defs[x].set_semi_rotameric(-90.0)

        ### detect if CHI rotates an amide group
        if c2.GetType() == 'C.3' and c3.GetType() == 'C.2' and c4.GetType() == 'N.am':
          nbrs = [atm.GetType() for atm in c3.GetAtoms() if atm not in [c1, c2, c3, c4]]
          if 'O.2' in nbrs:
            print('   *** identified CHI rotating amide group')
            if (x+1 == len(chi_defs)) and allow_semirotameric:
              print('   *** found terminal amide -- setting CHI to semi-rotameric')
              chi_defs[x].set_semi_rotameric(-180.0)

        if c3.GetType() == 'C.3' and c2.GetType() == 'C.2' and c1.GetType() == 'N.am':
          nbrs = [atm.GetType() for atm in c2.GetAtoms() if atm not in [c1, c2, c3, c4]]
          if 'O.2' in nbrs:
            print('   *** identified CHI rotating amide group')
            if (x+1 == len(chi_defs)) and allow_semirotameric:
              print('   *** found terminal amide -- setting CHI to semi-rotameric')
              chi_defs[x].set_semi_rotameric(-180.0)

        ### first chi for peptoids
        if PEPTOID and c2.GetType() == 'N.am':
          ### using guidance from Refrew et al, JACS 2014
          print('   * first chi for peptoid type -- adjusting to-90, 90')
          torlib[foundkey] = [-90.0, 90.0]

        ### temp for S.3 / C.3 torsion
        if (c2.GetType() == 'S.3' and c3.GetType() == 'C.3') or (c3.GetType() == 'S.3' and c2.GetType() == 'C.3') and (c1.GetType() != 'H') and (c4.GetType() != 'H'):
          print('   * adjusting chi values of S.3/C.3 torsion to -60, 60, 180')
          torlib[foundkey] = [-60.0, 60.0, 180.0]

        ### append list of torsions for chi, assuring angles in range of 0.0 to 360.0
        ### otherwise, remove chi_def from list
        if not skip:
          ### keep symm == 2 within 0 and 180
          ### TODO: this can use the wrap_angle function
          if chi_defs[x].symmetry == 2:
            # make sure there are no duplicates
            symm_torsions = list(set([x+180.0 if x < 0.0 else x for x in torlib[foundkey]]))
            if 0 in symm_torsions and 180 in symm_torsions:
              symm_torsions.remove(180)
            chi_defs[x].set_ideal_torsions(symm_torsions)
          ### keep symm == 3 within 0 and 120
          ### TODO: this can use the wrap_angle function
          elif chi_defs[x].symmetry == 3:
            symm_torsions_1 = []
            for tor in torlib[foundkey]:
              if tor > 120.0:
                tor -= 360.0
              while tor < 0.0:
                tor += 120.0
              symm_torsions_1.append(tor)
            symm_torsions = list(set(symm_torsions_1))
            if 0 in symm_torsions and 120 in symm_torsions:
              symm_torsions.remove(120)
            chi_defs[x].set_ideal_torsions(symm_torsions)
          else:
            chi_defs[x].set_ideal_torsions([x+360.0 if x < 0.0 else x for x in torlib[foundkey]])
      ### if torlib[foundkey] is not found, remove from list of paramerized
      else:
        del_list.append(x)

    ### regenerate chi_defs list removing indexes found in del_list
    chi_defs = [chi_defs[x] for x in range(len(chi_defs)) if x not in del_list]
    print('final number of chis in chi_defs = %i'%(len(chi_defs)))

    print('\nsetting phi/psi to %6.1f/%6.1f'%(phi,psi))

    ### cut bonds before setting phi/psi, just in case the cut bond cyclizes the backbone
    ### no need to check if the cut_bond actually 
    if len(cut_bonds) > 0:
      delete_cut_bonds( single_conf, cut_bonds, atom_map )

    single_conf.SetTorsion( bb1, bb2, bb3, bb4, phi*PI/180.0 )
    single_conf.SetTorsion( bb2, bb3, bb4, bb5, psi*PI/180.0 )
    ### set omega for N-cyclized cases
    single_conf.SetTorsion( bb0, bb1, bb2, bb3, omega*PI/180.0 )


    if len(cut_bonds) > 0:
      rebuild_cut_bonds( single_conf, cut_bonds, atom_map )

    ### initial cluster centroids -- used for binning of rotamers
    torsion_angles = [ chi.ideal_torsions for chi in chi_defs ]
    print('torsion_angles = ',torsion_angles)
    ideal_centroids = [Rotamer(x,None) for x in list(itertools.product(*torsion_angles))]
    print('\ntotal number of CHI bins = %i'%(len(ideal_centroids)))

    for x in range(len(ideal_centroids)):
      bins = get_bin_assignments(ideal_centroids[x].chis,torsion_angles)
      ### associate CHI combos to bin numbers here
      ideal_centroids[x].set_bins(bins)
      print('ideal centoids chis/bins: ',ideal_centroids[x].chis, ideal_centroids[x].bins)

    ### create list of lists for the sampling of each CHI
    if TEST_RUN:
      increment = 45
    else:
      if len(chi_defs) == 1:
        increment = 1
      elif len(chi_defs) == 2:
        increment = 5
      elif len(chi_defs) == 3:
        increment = 15
      elif len(chi_defs) == 4:
        increment = 30

    tor_list = [x for x in range(0,360,increment)]
    tor_test = [tor_list for x in range(len(chi_defs))]
    ### create all combination of initial CHI sampling to minimize
    all_rot_test = list(itertools.product(*tor_test))

    ### append NCAA_ROTLIB_PATH and NCAA_ROTLIB_NUM_ROTAMER_BINS info
    ### to end of XXX.params file -- only do this once!
    if not PARAMS_BINS:
      with open('%s.params'%(aa_name),'a') as params_file:
        params_file.write('NCAA_ROTLIB_PATH %s.rotlib\n'%(aa_name))
        if len(chi_defs) > 0:
          params_file.write('NCAA_ROTLIB_NUM_ROTAMER_BINS %i '%(len(chi_defs)))
          for x in range(len(chi_defs)):
            params_file.write('%i '%(len(chi_defs[x].ideal_torsions)))




    ### minimize initial rotamers
    count = 0
    start_chis = []; initial_rotamers = []
    print('generating %i conformers'%(len(all_rot_test)))
    for rot_test in all_rot_test:
      count += 1
      if len(proton_chi_atom_names) > 0:

        ### set chi angles of single_conf prior to mimization
        working_mol = oechem.OEMol(mol.SCMol())
        working_mol.DeleteConfs()
        sz_results = OESzybkiResults()
        for x in range(len(rot_test)):
          a1 = chi_atom_names[x][0]; a2 = chi_atom_names[x][1]; a3 = chi_atom_names[x][2]; a4 = chi_atom_names[x][3]
          a1a = single_conf.GetAtom( OEHasAtomName(a1) )
          a2a = single_conf.GetAtom( OEHasAtomName(a2) )
          a3a = single_conf.GetAtom( OEHasAtomName(a3) )
          a4a = single_conf.GetAtom( OEHasAtomName(a4) )
          sc_chi = single_conf.GetTorsion(a1a,a2a,a3a,a4a)
          single_conf.SetTorsion( single_conf.GetAtom( OEHasAtomName(a1) ), single_conf.GetAtom( OEHasAtomName(a2) ), single_conf.GetAtom( OEHasAtomName(a3) ), single_conf.GetAtom( OEHasAtomName(a4) ), rot_test[x]*PI/180.0 )
          sc_chi = single_conf.GetTorsion(a1a,a2a,a3a,a4a)
        starting_chis = get_chis( single_conf, chi_defs, atom_map, symm=False )

        # now scan proton chis
        min_energy = 1e6
        ### TODO: this is currently only functional for a single proton CHI
        ### in the case of multiple proton chis, create an exhaustive list of combiniations
        ### similar to heavy atom CHIS and apply/min all of these
        for y in range(len(proton_chi_atom_names)):
          for proton_angle in range(-180,180,60):
            mol_copy = OEMol( single_conf.CreateCopy() )
            working_conf = mol_copy.GetActive()

            pa1 = working_conf.GetAtom( OEHasAtomName( proton_chi_atom_names[y][0] ) )
            pa2 = working_conf.GetAtom( OEHasAtomName( proton_chi_atom_names[y][1] ) )
            pa3 = working_conf.GetAtom( OEHasAtomName( proton_chi_atom_names[y][2] ) )
            pa4 = working_conf.GetAtom( OEHasAtomName( proton_chi_atom_names[y][3] ) )
            working_conf.SetTorsion( pa1, pa2, pa3, pa4, proton_angle*PI/180.0 )
            p_chi = working_conf.GetTorsion(pa1,pa2,pa3,pa4)*180/PI
            starting_chis = get_chis( working_conf, chi_defs, atom_map, symm=True )
            starting_chis[-1] = p_chi
            sz(working_conf,sz_results)
            ending_chis = get_chis( working_conf, chi_defs, atom_map, symm=True )
            p_chi = working_conf.GetTorsion(pa1,pa2,pa3,pa4)*180/PI
            ending_chis[-1] = p_chi
            if working_conf.GetEnergy() < min_energy:
              min_energy = working_conf.GetEnergy()
              min_conf = OEMol( working_conf.CreateCopy() )
        ending_chis = get_chis( min_conf, chi_defs, atom_map, symm=True )
        initial_rotamers.append(Rotamer(list(ending_chis[:-1]),ending_chis[-1]))
        initial_rotamers[-1].set_conf(OEMol( min_conf ))
        initial_rotamers[-1].set_initial_chis(list(rot_test))
        del mol_copy
        del working_conf
        del min_conf

      else:
        ### before setting torsions, delete cut_bonds if present
        if len(cut_bonds) > 0:
          delete_cut_bonds( single_conf, cut_bonds, atom_map )

        for x in range(len(rot_test)):
          single_conf.SetTorsion( single_conf.GetAtom( OEHasAtomName(chi_atom_names[x][0]) ), single_conf.GetAtom( OEHasAtomName(chi_atom_names[x][1]) ), single_conf.GetAtom( OEHasAtomName(chi_atom_names[x][2]) ), single_conf.GetAtom( OEHasAtomName(chi_atom_names[x][3]) ), rot_test[x]*PI/180.0 )
        starting_chis = get_chis( single_conf, chi_defs, atom_map, symm=False )
        sz_results = OESzybkiResults()

        ### before minimizing, rebuild cut_bonds
        if len(cut_bonds) > 0:
          rebuild_cut_bonds( single_conf, cut_bonds, atom_map )

        ### energy minimize test conformation

        if sz(single_conf,sz_results):
          ending_chis = get_chis( single_conf, chi_defs, atom_map, symm=True )
          start_chis.append(Rotamer(list(starting_chis[:-1]),None))
          initial_rotamers.append(Rotamer(list(ending_chis[:-1]),ending_chis[-1]))
          initial_rotamers[-1].set_conf(OEMol( single_conf ))
          initial_rotamers[-1].set_initial_chis(list(rot_test))
        else:
          print('minimzation failed for %f/%f'%(rot_test[0],rot_test[1]))

    ### calculate distance between all minimized rotamers for clustering
    print('calculating all distances')
    calc_all_dist( initial_rotamers, ideal_centroids, chi_defs )
    print('assigning cluster numbers')
    assign_cluster_number( initial_rotamers )
    print('calculating boltzmann weights')
    calc_boltzmann_weights( initial_rotamers )

    ### replace ideal_centroids with centroids of initial clusters
    for x in range(2):
      print('\nk-means clustering iteration %i'%(x+1))
      updated_centroids = calc_cluster_centroids( initial_rotamers, chi_defs )
      calc_all_dist( initial_rotamers, updated_centroids, chi_defs )
      assign_cluster_number( initial_rotamers )
      calc_boltzmann_weights( initial_rotamers )

    ### find low-energy rotamer for each cluster and sort by probability
    final_rotamers = get_final_rotamers( initial_rotamers, ideal_centroids )
    final_rotamers.sort(key=attrgetter('boltzmann_weight'),reverse=True)

    ### get std_devs and print single phi/psi rotlib for Rosetta
    fout = open('%s_%i_%i_%i.rotlib'%(aa_name,int(phi),int(psi),int(omega)),'w')

    ### get std_devs and print single phi/psi rotlib for Rosetta
    rot_count = 0
    for rot in final_rotamers:
      rot_count += 1

      if TEST_RUN and rot.conf:
        ### dump final conformere
        ofs = oemolostream()
        ofs.open('%s_final_rot_%i.mol2'%(aa_name,rot_count))
        print('printing conformer for rotamer: %2i -- '%(rot_count),rot.conf)
        OEWriteMolecule(ofs,rot.conf)

      ### get standard deviations for final rotamers
      if rot.boltzmann_weight == 0:
        ### if boltzmann_weight = 0, then this conformation is either extremly improbable
        ### or unassigned. either way, set std_devs to maximal value of 30.0
        rot.set_stds( pad_list_with_zeros([30]*len(chi_defs),4) )
      else:
        rot.set_stds( get_std_devs( rot.conf, rot.chis, chi_defs, cut_bonds, cut_bonds_2, atom_map ) ) 

      ### expand rotamer.chis to list of length 4  for Rosetta formatting  
      rot.set_chis( pad_list_with_zeros(rot.chis,4) )


    ### print single phi/psi rotlib final data for Rosetta
    ### perform semi_rotameric averaging if necessary
    ###
    ### TODO: CODE DUPLICATION -- CLEAN THIS UP BELOW

    if chi_defs[-1].semi_rotameric:
      chi_bin_dict = defaultdict(list)
      for rot in final_rotamers:
        if rot.boltzmann_weight == 0:
          continue
        else:
          rotkey = ''.join(map(str,rot.bins))[:len(chi_defs)-1]
          print('rotkey = ',rotkey)
          chi_bin_dict[rotkey].append( SemiRot(rot.chis[:len(chi_defs)-1],rot.boltzmann_weight,rot.stds[:len(chi_defs)-1]) )

      final_semi_rots = {}
      for key in chi_bin_dict:

        final_rot_1 = SemiRot( list(), 0.0, list() )
        final_rot_1.set_bins(list(key))

        prob_list = [rot.prob for rot in chi_bin_dict[key]]
        final_rot_1.set_probability( sum(prob_list) )

        for x in range(len(key)):
          chi_list  = [rot.chis[x] for rot in chi_bin_dict[key]]
          std_list  = [rot.stds[x] for rot in chi_bin_dict[key]]
          prob_list = [rot.prob for rot in chi_bin_dict[key]]
          final_rot_1.append_chi( circ_mean( chi_list ) )
          final_rot_1.append_std( average( std_list ) )
        final_semi_rots[key] = final_rot_1

      ### print output assuring that all rotameric CHI values are the same for all rotameric CHI bins
      for rot in final_rotamers:
        print('final_rotamer chis = ',rot.chis,'; bins = ',rot.bins)
        rotkey = ''.join(map(str,rot.bins))[:len(chi_defs)-1]
        if len(rotkey) == 0:
          fout.write('%s  %4i %4i %4i  %2i %2i %2i %2i  %10.8f  %6.1f %6.1f %6.1f %6.1f  %5.1f %5.1f %5.1f %5.1f\n'%('UNK', phi, psi, omega, rot.bins[0], rot.bins[1], rot.bins[2], rot.bins[3], rot.boltzmann_weight, periodic_range(rot.chis[0]), periodic_range(rot.chis[1]), periodic_range(rot.chis[2]), periodic_range(rot.chis[3]), rot.stds[0], rot.stds[1], rot.stds[2], rot.stds[3]))
        elif len(rotkey) == 1:
          fout.write('%s  %4i %4i %4i  %2i %2i %2i %2i  %10.8f  %6.1f %6.1f %6.1f %6.1f  %5.1f %5.1f %5.1f %5.1f\n'%('UNK', phi, psi, omega, rot.bins[0], rot.bins[1], rot.bins[2], rot.bins[3], rot.boltzmann_weight, periodic_range(final_semi_rots[rotkey].chis[0]), periodic_range(rot.chis[1]), periodic_range(rot.chis[2]), periodic_range(rot.chis[3]), final_semi_rots[rotkey].stds[0], rot.stds[1], rot.stds[2], rot.stds[3]))
        elif len(rotkey) == 2:
          fout.write('%s  %4i %4i %4i  %2i %2i %2i %2i  %10.8f  %6.1f %6.1f %6.1f %6.1f  %5.1f %5.1f %5.1f %5.1f\n'%('UNK', phi, psi, omega, rot.bins[0], rot.bins[1], rot.bins[2], rot.bins[3], rot.boltzmann_weight, periodic_range(final_semi_rots[rotkey].chis[0]), periodic_range(final_semi_rots[rotkey].chis[1]), periodic_range(rot.chis[2]), periodic_range(rot.chis[3]), final_semi_rots[rotkey].stds[0], final_semi_rots[rotkey].stds[1], rot.stds[2], rot.stds[3]))
        elif len(rotkey) == 3:
          fout.write('%s  %4i %4i %4i  %2i %2i %2i %2i  %10.8f  %6.1f %6.1f %6.1f %6.1f  %5.1f %5.1f %5.1f %5.1f\n'%('UNK', phi, psi, omega, rot.bins[0], rot.bins[1], rot.bins[2], rot.bins[3], rot.boltzmann_weight, periodic_range(final_semi_rots[rotkey].chis[0]), periodic_range(final_semi_rots[rotkey].chis[1]), periodic_range(final_semi_rots[rotkey].chis[2]), periodic_range(rot.chis[3]), final_semi_rots[rotkey].stds[0], final_semi_rots[rotkey].stds[1], final_semi_rots[rotkey].stds[2], rot.stds[3]))
        else:
          sys.exit('ERROR: found more than three rotameric CHIs!')

      fout.close()
    else:
      ### print formatted rotlib data for non-semirotameric aa
      for rot_num, rot in enumerate(final_rotamers):
        if PEPTOID:
          fout.write('%s  %4i  %4i  %4i  9999  %2i %2i %2i %2i  %10.8f  %6.1f %6.1f %6.1f %6.1f  %5.1f %5.1f %5.1f %5.1f\n'%('UNK', omega, phi, psi, rot.bins[0], rot.bins[1], rot.bins[2], rot.bins[3], rot.boltzmann_weight, periodic_range(rot.chis[0]), periodic_range(rot.chis[1]), periodic_range(rot.chis[2]), periodic_range(rot.chis[3]), rot.stds[0], rot.stds[1], rot.stds[2], rot.stds[3]))
        else:
          fout.write('%s  %4i %4i 9999  %2i %2i %2i %2i  %10.8f  %6.1f %6.1f %6.1f %6.1f  %5.1f %5.1f %5.1f %5.1f\n'%('UNK', phi, psi, rot.bins[0], rot.bins[1], rot.bins[2], rot.bins[3], rot.boltzmann_weight, periodic_range(rot.chis[0]), periodic_range(rot.chis[1]), periodic_range(rot.chis[2]), periodic_range(rot.chis[3]), rot.stds[0], rot.stds[1], rot.stds[2], rot.stds[3]))

        if len(chi_defs) == 1:
          print('final rotamer %2i: X1 = %7.2f, energy = %7.2f, probability = %7.3f'%(rot_num+1, periodic_range(rot.chis[0]), rot.energy, 100*rot.boltzmann_weight))
        if len(chi_defs) == 2:
          print('final rotamer %2i: X1 = %7.2f, X2 = %7.2f, energy = %7.2f, probability = %7.3f'%(rot_num+1, periodic_range(rot.chis[0]), periodic_range(rot.chis[1]), rot.energy, 100*rot.boltzmann_weight))
        if len(chi_defs) == 3:
          print('final rotamer %2i: X1 = %7.2f, X2 = %7.2f, X3 = %7.2f, energy = %7.2f, probability = %7.3f'%(rot_num+1, periodic_range(rot.chis[0]), periodic_range(rot.chis[1]), periodic_range(rot.chis[2]), rot.energy, 100*rot.boltzmann_weight))
      fout.close()

    ###
    ### perform semi-rotameric check and calculation
    ###
    if chi_defs[-1].semi_rotameric:

      ### get semi-rotameric range based on start angle (determined during bin assignments) and symmetry
      ### i think chi_defs.symmetry == 3 should never be semi_rotameric
      print('\n\nperform semi_rotameric calculation on terminal CHI, starting at %f'%(chi_defs[-1].semi_rotameric))
      start_val = chi_defs[-1].semi_rotameric
      if chi_defs[-1].symmetry == 2:
        end_val = start_val+180.0
        semi_rot_step = 5.0
      else:
        end_val = start_val+360.0
        semi_rot_step = 10.0

      semirange = arange(start_val,end_val,semi_rot_step)
      print('semi-rotameric range = ',semirange)

      ### get average probability for nchi-1 bin groupings
      for x in range(len(chi_defs)-1):
        print(x,len(chi_defs[x].ideal_torsions))

      chi_bin_dict = defaultdict(list)
      for rot in final_rotamers:
        if rot.boltzmann_weight == 0:
          continue
        else:
          print(rot.bins, rot.chis, rot.boltzmann_weight) 
          rotkey = ''.join(map(str,rot.bins))[:len(chi_defs)-1]
          chi_bin_dict[rotkey].append( SemiRot(rot.chis[:len(chi_defs)-1],rot.boltzmann_weight,rot.stds[:len(chi_defs)-1]) )

      all_probs = []
      final_rots = []
      for key in chi_bin_dict:

        final_rot_1 = SemiRot( list(), 0.0, list() )
        final_rot_1.set_bins(list(key))

        print('key = %s'%(key))
        prob_list = [rot.prob for rot in chi_bin_dict[key]]
        print('  sum of probabilities = %f'%(sum(prob_list)))
        all_probs.append(sum(prob_list))
        final_rot_1.set_probability( sum(prob_list) )

        for x in range(len(key)):
          chi_list  = [rot.chis[x] for rot in chi_bin_dict[key]]
          std_list  = [rot.stds[x] for rot in chi_bin_dict[key]]
          prob_list = [rot.prob for rot in chi_bin_dict[key]]
          print('   average chi %7.2f ( %7.2f )'%(circ_mean(chi_list),std(chi_list)))
          final_rot_1.append_chi( circ_mean( chi_list ) )
          final_rot_1.append_std( average( std_list ) )

        all_chis = [rot.chis for rot in chi_bin_dict[key]]

        prob_dist_list = []
        for rot in chi_bin_dict[key]:
          prob_dist = get_semirotameric( mol, rot.chis, chi_defs, atom_map, semirange )
          prob_dist_list.append(prob_dist)
          print('%.8f'%rot.prob, ['%.2f'%x for x in rot.chis], '%.2f'%(min(prob_dist)), ['%.2f'%x for x in prob_dist][:10])
          ### take minimum energy at each semirange value for each individual torsion scan

        min_energy_dist = [ min([prob_list[x] for prob_list in prob_dist_list]) for x in range(len(semirange)) ]          
        prob_dist = boltzmann_dist( min_energy_dist )        

        final_rot_1.set_prob_dist( min_energy_dist )
        print('%.8f'%final_rot_1.probability, ['%.2f'%x for x in final_rot_1.chis], '%.2f'%(min(final_rot_1.prob_dist)), ['%.2f'%x for x in final_rot_1.prob_dist][:10])
        final_rot_1.set_prob_dist( prob_dist )      
        final_rots.append(final_rot_1)

      print('sum of semi-rot bin probs = %f'%(sum(all_probs)))

      ### sort final semi-rotameric rotamers by probability and print file
      final_rots.sort(key=attrgetter('probability'),reverse=True)

      fout = open('%s_%i_%i.densities.rotlib'%(aa_name,int(phi),int(psi)),'w')
      for rot in final_rots:
        fout.write('%s %4i %4i 9999 '%('UNK', phi, psi ))
        for bin_val in rot.bins:
          fout.write('%2s  '%(bin_val))  
        fout.write('%8.6f  '%(rot.probability))
        for chi in rot.chis:
          fout.write('%6.1f  '%(periodic_range(chi)))
        for std_dev in rot.stds:
          fout.write('%4.1f  '%(std_dev))
        for prob in rot.prob_dist:
          fout.write('%.4g '%(prob))
        fout.write('\n')
      fout.close()

      if chi_defs[-1].semi_rotameric:
        input_data['semi-rot'] = open('%s_%i_%i_%i.densities.rotlib'%(aa_name,int(phi),int(psi),int(omega))).read()

  return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))

