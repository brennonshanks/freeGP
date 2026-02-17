""" UNIT CONVERSION FACTORS AND OTHER CONSTANTS """

k_B_kcal = 1.987204e-3     # Boltzmann constant in kcal/(mol*K)
k_B      = 8.3144621e-3    # Boltzmann constant in kJ/(mol*K)
T        = 303.15            # Temperature in Kelvin
beta = 1/(k_B*T)      # kcal/mol (or kJ/mol)

# NOTE: be careful about the units! Aim for kJ/mol or kT. Fuck the kcal/mol.
# Katka's forces are supposedly written in kJ/mol/nm in the 'pullf' files.
# The force constants we are taking from the .mdp files are in kJ/mol/nm^2 and we utilize only
# the 'pullx' files. After a careful dimensional analysis, we definitely should use kJ/mol everywhere.
