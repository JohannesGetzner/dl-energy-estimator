#!/bin/bash
# This script is needed to change the access rights of Intel's RAPL files 
# such that they can be accessed by the different energy profiling tools.

sudo chmod 555 /sys/class/powercap/intel-rapl/intel-rapl\:0/energy_uj
sudo chmod 555 /sys/class/powercap/intel-rapl/intel-rapl\:0/max_energy_range_uj

sudo chmod 555 /sys/class/powercap/intel-rapl/intel-rapl\:0/intel-rapl\:0\:0/energy_uj
sudo chmod 555 /sys/class/powercap/intel-rapl/intel-rapl\:0/intel-rapl\:0\:0/max_energy_range_uj

sudo chmod 555 /sys/class/powercap/intel-rapl/intel-rapl\:0/intel-rapl\:0\:1/energy_uj
sudo chmod 555 /sys/class/powercap/intel-rapl/intel-rapl\:0/intel-rapl\:0\:1/max_energy_range_uj

sudo chmod 555 /sys/class/powercap/intel-rapl/intel-rapl\:0/intel-rapl\:0\:2/energy_uj
sudo chmod 555 /sys/class/powercap/intel-rapl/intel-rapl\:0/intel-rapl\:0\:2/max_energy_range_uj


sudo chmod 555 /sys/class/powercap/intel-rapl/intel-rapl:0
sudo chmod 555 /sys/class/powercap/intel-rapl/intel-rapl:1
sudo chmod 555 /sys/class/powercap/intel-rapl/intel-rapl:1/name

sudo chmod 555 -R /sys/class/powercap/intel-rapl