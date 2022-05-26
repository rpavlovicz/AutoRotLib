# AutoRotLib

** note: these script call on OpenEye tools, particularly
  openeye.oechem
  openeye.oeomega
  openeye.oequacpac
  openeye.oeszybki
** thus a valid OpenEye license for these tools is required

1.) create a SMILES string of your amino acid that has an acetyl group added on to the N-terminus and an N-methyl group added to the C-terminal end

aspartic acid example:

>OC(NC)[C@H](CC(=O)[O-])NC(C)=O

2.) run the "charge_and_correct.py" script, providing the following inputs:
  text file with the amino acid SMILES string
  total charge of side chain
  three letter identifier

  >python charge_and_correct.py --smiles test.smi --charge -1 --aa_name ASP &> ASP_torlib_OE

  this will generate a mol2 file with partial atomic charges and some footers needed for step 3
  also, pipe the output to a file to be used as input to step 4

3.) run molfile_to_params_polymer_ARL.py to create .params file for amino acid

  >python molfile_to_params_polymer.py --clobber --polymer --no-pdb --extra_torsion_output --name ASP ./ASP_AM1BCC.mol2

  this generates the .params file that is used on the Rosetta command line to load a new amino acid topology
  it will also generate a XXX.atom_map file needed for step 4
  you may use the --extra_torsion_output file to generate constraints for the side chain depending on your intended use in Rosetta

4.) run the make_rotlib_phi_psi.py script for each phi/psi combination

  this requires the ASP_torlib_OE file generated in step 2 in addition to other outputs from previuos steps so it is important to maintain
  the same 3-letter code for your amino acid

  >python make_rotlib_phi_psi.py -n ASP --phi -180 --psi 180

  this script will also add two lines to the end of your .params file adding information about the number of idealized chi values
  in addition to the location of the final rotamer library

5.) once all of the individual rotamer library files are generated for each phi/psi combinationn, combine them all into the final
rotamer library with the "make_final_rotlib.py" script

  >python make_final_rotlib.py -n ASP
