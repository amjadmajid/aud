#!/bin/bash

#script to run playrec.sh multiple times for all
# music files in the folder /Music_files

#./playrec.sh --distance 10 --direction 0 --top y --music W_noise_32s --location T -t

# help message
function show_usage (){
  printf "Usage: $0 [options [parameters]]\n"
  printf "\n"
  printf "Options:\n"
  printf " --distance  : Distance to sound source in cm\n"
  printf " --direction : Angle to sound source in degrees\n"
  printf " --los       : Defines if there is line of sight [y/n], default y\n"
  printf " --top       : Defines if the top of the mic array is obstructed [y/n], default n\n"
  printf " --edistance : Euclidian distance to source in cm(only if LoS == n)\n"
  printf " --edirection: Euclidian angle to source in degrees(only if LoS == n)\n"
  printf " --location  : Denote location of recording"
  printf " -t          : If this argument is given, then no sounds will be played or recorded (test mode)"
  printf " -h|--help   : Print this\n"

return 0
}

args=
loc=

# Argument parsing
while [ ! -z "$1" ]; do
  case "$1" in
    --distance)
      shift
      args="$args --distance $1"
      ;;
    --direction)
      shift
      args="$args --direction $1"
      ;;
    --los)
      shift
      args="$args --los $1"
      ;;
    --top)
      shift
      args="$args --top $1"
      ;;
    --edistance)
      shift
      args="$args --edistance $1"
      ;;
    --edirection)
      shift
      args="$args --edirection $1"
      ;;
    --location)
      shift
      loc=$1
      ;;
    -t)
      args="$args -t"
      ;;
    *)
      show_usage
      ;;
  esac

  if [ $# -gt 0 ]; then
	  shift
  fi
done

#make list of music files
musicloc="Music_files/"
extension=".wav"


for j in 1 2; do #repeat experiment twice 
  for i in $musicloc*$extentsion; do
    [ -f "$i" ] || continue   #guards for a case where ther are no existing files
    musicname=${i#"$musicloc"}
    musicname=${musicname%"$extension"}
    echo $musicname
    echo $args --location $loc$j
    ./playrec.sh $args --music $musicname --location $loc$j
  
  done
done
