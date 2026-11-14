'''
This file is used to run multiple bde scans, one after the other.

Run this file with 'python run_many_scans.py'
'''

import bde_scan
import send_email

#  The format for each entry in the array looks like this:
#  [NAME, EXP_BDE_KCAL, SMILES_STRING, BOND_TO_SCAN, SCAN_NUMBER, INDICES_OF_BOND, USE_GPAW, EXTEND_GRAPH, HYPER_EXTEND_GRAPH, DO_RIGID]
#  For instructions on what each of these do, check the main python file.

SEND_EMAIL = False # If True, confirmation email is sent upon completion with the following data:
process_name = "" # Name of the process to be included in the email

'''
scans_to_run = [
    # ALKANES
    ["Ethane", 101.1, "CC", "C-H", 0, [0, 2], False, False, True],
    ["Ethane", 90.1, "CC", "C-C", 0, [0, 1], False, False, True],

    # ALCOHOLS
    ["Ethanol", 94.5, "CCO", "C-H", 0, [0, 4], False, False, True], 
    ["Ethanol", 84.5, "CCO", "C-C", 0, [0, 1], False, False, True],
    ["Ethanol", 92.5, "CCO", "C-O", 0, [1, 2], False, False, True],

    # AMINES
    ["Ethylamine", 91.5, "CCN", "C-H", 0, [0, 4], False, False, True],
    ["Ethylamine", 82.5, "CCN", "C-C", 0, [0, 1], False, False, True],
    ["Ethylamine", 103.0, "CCN", "N-H", 0, [2, 8], False, False, True],
    ["Ethylamine", 84.5, "CCN", "C-N", 0, [1, 2], False, False, True],

    # NITRO COMPOUNDS
    ["Nitroethane", 97.5, "CC[N+]([O-])=O", "C-H", 0, [0, 7], False, False, True],
    ["Nitroethane", 81.0, "CC[N+]([O-])=O", "C-C", 0, [0, 1], False, False, True],
    ["Nitroethane", 59.4, "CC[N+]([O-])=O", "C-N", 0, [1, 2], False, False, True],
    ["Nitroethane", 60.5, "CC[N+]([O-])=O", "N-O", 0, [2, 3], False, False, True],

    # PEROXIDES
    ["Ethyl-Hydroperoxide", 95.0, "CCOO", "C-H", 0, [0, 6], False, False, True],
    ["Ethyl-Hydroperoxide", 81.5, "CCOO", "C-C", 0, [0, 1], False, False, True],
    ["Ethyl-Hydroperoxide", 44.0, "CCOO", "O-O", 0, [2, 3], False, False, True],

    # CARBONYLS
    ["Methanal", 87.2, "C=O", "C-H", 0, [0, 2], False, False, True],
    ["Methanal", 179.0, "C=O", "C-O", 0, [0, 1], False, False, True]
]


scans_to_run = [
    # Running relaxed dissociation scan on some products of Nitromethane Decomposition
    ["Methyl-Nitrite", 41.7, "CON=O", "O-N", 0, [1, 2], False, False, True],
    ["Methoxy-Radical", 101.0, "[O]C", "C-H", 0, [1, 2], False, False, True],
    ["Formyl-Radical", 18.0, "[CH]=O", "C-H", 0, [0, 1], False, False, True]
]
'''

scans_to_run = [
    # Running relaxed dissociation scans (rigid & relaxed) to investigate the peroxide O-O bond failure

    # Rigid
    ["Hydrogen-Peroxide", 50.0, "OO", "O-O", "rigid", [0, 1], False, True, False, True],
    ["Ethyl-Hydroperoxide", 41.6, "CCOO", "O-O", "rigid", [2, 3], False, False, True, True],
    ["Dioxygen-Difluoride", 80.0, "FOOF", "O-O", "rigid", [1, 2], False, True, False, True],
    ["Dichlorine-Peroxide", 18.0, "ClOOCl", "O-O", "rigid", [1, 2], False, True, False, True],
    ["Diatomic-Oxygen", 119.2, "O=O", "O=O", "rigid", [0, 1], False, False, False, True],
    # Relaxed
    ["Hydrogen-Peroxide", 50.0, "OO", "O-O", "relaxed", [0, 1], False, True, False, False],
    ["Ethyl-Hydroperoxide", 41.6, "CCOO", "O-O", "relaxed", [2, 3], False, False, True, False],
    ["Dioxygen-Difluoride", 80.0, "FOOF", "O-O", "relaxed", [1, 2], False, True, False, False],
    ["Dichlorine-Peroxide", 18.0, "ClOOCl", "O-O", "relaxed", [1, 2], False, True, False, False]
]

for scan in scans_to_run:
    
    success = bde_scan.run(scan[0], scan[1], scan[2], scan[3], scan[4], scan[5], scan[6], scan[7], scan[8], scan[9])
    
    if not success:
        print(f"\nThe following scan failed (Ensure you provided the correct values): {scan}\n")

if SEND_EMAIL:
    send_email.send_completion_email(process_name)
