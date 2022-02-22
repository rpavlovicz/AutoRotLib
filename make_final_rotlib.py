#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os,sys
import fileinput
import argparse
from glob import glob
from collections import defaultdict

def check_arg(args=None):
  parser = argparse.ArgumentParser(description="combine N input files for MakeRotLib, where N = n_phi_bins*n_psi_bins")
  parser.add_argument('-n', '--aa_name', required=True, type=str, help='three letter name for residue')
  parser.add_argument('--peptoid', required=False, type=bool, default=False, help='gather samples +/- 30 degrees from cis and trans omega angles')
  results = parser.parse_args(args)
  return (results.aa_name, results.peptoid)

aa, PEPTOID = check_arg(sys.argv[1:])

logs = glob('%s_*.rotlib'%(aa))

rotlib_dict = defaultdict(list)

for log in logs:
  for line in fileinput.input(log):
    if len(line.rsplit()) == 18 and not PEPTOID:
      sys.exit('detected non-standard output. use --peptoid flag?')
    if PEPTOID:
      omega = int(line.rsplit()[1])
      phi = int(line.rsplit()[2])
      psi = int(line.rsplit()[3])
      rotlib_dict[(omega,phi,psi)].append(line)
    else:
      phi = int(line.rsplit()[1])
      psi = int(line.rsplit()[2])
      rotlib_dict[(phi,psi)].append(line)

fout = open('%s.rotlib'%(aa),'w')

if PEPTOID:
  count = 0
  duplicate_count = 0
  omega_list = [150,160,170,180,-170,-160,-150,-30,-20,-10,0,10,20,30]
  for phi in range(-180,180+10,10):
    for psi in range(-180,180+10,10):
      for omega in omega_list:
        if len(rotlib_dict[(omega,phi,psi)]) == 0:
          sys.exit('missing data for omega = %i; phi = %i; psi = %i'%(omega,phi,psi))
        if len(rotlib_dict[(omega,phi,psi)]) != len(list(set(rotlib_dict[(omega,phi,psi)]))):
          duplicate_count += 1
          print('%5i -- duplicates found for %i,%i,%i'%(duplicate_count,omega,phi,psi))
          for x in rotlib_dict[(omega,phi,psi)]:
            print('  ',x)
        added = []
        count += 1
        for line in rotlib_dict[(omega,phi,psi)]:
          if line in added:
            continue
          else:
            fout.write(line)
            added.append(line)
  print('final count = %i'%(count)) 

else:
  for phi in range(-180,180+10,10):
    for psi in range(-180,180+10,10):
      if len(rotlib_dict[(phi,psi)]) == 0:
        sys.exit('missing data for phi = %i; psi = %i'%(phi,psi))
      if len(rotlib_dict[(phi,psi)]) != len(list(set(rotlib_dict[(phi,psi)]))):
          print('duplicates found for %i,%i'%(phi,psi))
      added = []
      for line in rotlib_dict[(phi,psi)]:
        if line in added: 
          continue
        else:
          fout.write(line)
          added.append(line)

fout.close()

