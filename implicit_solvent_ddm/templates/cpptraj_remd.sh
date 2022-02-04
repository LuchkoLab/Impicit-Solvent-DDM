#!/bin/tcsh

cpptraj $solute << EOF
trajin $trajectory remdtraj remdtrajtemp $target_temperature
trajout $temperature_traj.nc nobox

go

EOF