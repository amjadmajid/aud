#!/bin/bash

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
  printf " --music     : name of music file located in the folder 'Music_files/' or '_no_music' for recording background noise"  
  printf " --location  : Denote location of recording"
  printf " -t          : If this argument is given, then no sounds will be played or recorded (test mode)"
  printf " -h|--help   : Print this\n"

return 0
}

dist=
dir=
LoS="y"
LoS_state="Line_of_Sight"
top="n"
top_state="Free_Top"
edist=
edir=
musicfile=
loc=
no_music_mode=false
test_mode=false

# Argument parsing
while [ ! -z "$1" ]; do
  case "$1" in
    --distance)
      shift
      dist=$1
      ;;
    --direction)
      shift
      dir=$1
      ;;
    --los)
      shift
      LoS=$1
      if [ "$LoS" == "y" ]; then
        LoS_state="Line_of_Sight"
      elif [ "$LoS" == "n" ]; then
        LoS_state="Non_Line_of_Sight"
      else
        echo "$LoS is an invalid argument for --los"
        show_usage
      fi
      ;;
    --top)
      shift
      top=$1
      if [ "$top" == "y" ]; then
        top_state="Obstructed_Top"
      elif [ "$top" == "n" ]; then
        top_state="Free_Top"
      else
        echo "$top is an invalid argument for --top"
        show_usage
      fi
      ;;
    --edistance)
      shift
      edist=$1
      ;;
    --edirection)
      shift
      edir=$1
      ;;
    --music)
      shift
      musicfile=$1
      if [ "$musicfile" == "_no_music" ]; then
		    no_music_mode=true
	    fi
			  
	    if [[ ! -f "Music_files/$musicfile.wav" ]]; then
	      echo "ERROR: Music_files/$musicfile.wav does not exist."
		  exit 1
	    fi
      ;;
    --location)
      shift
      loc=$1
      ;;
    -t)
      test_mode=true
      ;;
    *)
      show_usage
      ;;
  esac

  if [ $# -gt 0 ]; then
	  shift
  fi
done

# Test if all required arguments are present

if [ -z "$musicfile" ]; then
  echo "ERROR: music not set"
  exit 1
fi
    
if [ -z "$loc" ]; then
  echo "ERROR: location not set"
  exit 1
fi
  
if ! $no_music_mode ; then
  if [ -z "$dist" ]; then
    echo "ERROR: distance not set"
    exit 1
  fi

  if [ -z "$dir" ]; then
    echo "ERROR: direction not set"
    exit 1
  fi

  if [ "$LoS" == "n" ]; then
    if [ -z "$edist" ]; then
      echo "ERROR: Euclidian distance not set for non line of sight"
      exit 1
    fi

    if [ -z "$edir" ]; then
      echo "ERROR: Euclidian direction not set for non line of sight"
      exit 1
    fi
  else
    if [ -n "$edist" ]; then
      echo "WARNING: Euclidian distance is ignored if LoS = \"y\""
      echo $edist
    fi
  
    if [ -n "$edir" ]; then
      echo "WARNING: Euclidian direction is ignored if LoS = \"y\""
    fi

  fi
fi

# recording file name
if ! $no_music_mode ; then
  printf -v recfile "rec_%03dcm_%03d" $dist $dir
  if [ "$LoS" == "n" ]; then
    printf -v recfile "%s_euclid_%03dcm_%03d" $recfile $edist $edir
  fi
else
  recfile="Background"
fi
recfile="${recfile}_loc${loc}"

# make folder to save the recorded files
if ! $no_music_mode ; then
  recloc="Recorded_files/$top_state/$LoS_state/$musicfile/Raw_recordings"
else
  recloc="Recorded_files/$top_state/Background/Raw_recordings"
fi

if [[ ! -d $recloc ]]; then
  mkdir -p $recloc
fi

recloc="$recloc/$recfile.wav"
if ! $no_music_mode ; then
  musicloc="Music_files/$musicfile.wav"
fi

echo "------------------------------------------"
if $test_mode; then
  echo "TEST MODE"
fi
if ! $no_music_mode ; then
  echo "Music file name:        $musicfile.wav"
  echo "Music file location:    $musicloc"
  echo "Recorded file name:     $recfile.wav"
  echo "Recorded file location: $recloc"
  echo "Distance:               $dist cm"
  echo "Direction:              $dir degrees"
  echo "Top state:              $top_state"
  echo "Line of Sight state:    $LoS_state"
  
  if [ "$LoS" == "n" ]; then
    echo "Euclidian distance:     $edist cm"
    echo "Euclidian direction:    $edir degrees"
  fi  
  
  echo "Location:               loc$loc"
else
  echo "Recording background"
  echo "Recorded file name:     $recfile.wav"
  echo "Recorded file location: $recloc"
  echo "Top state:              $top_state"
  echo "Location:               loc$loc"
fi
echo "------------------------------------------"

if $test_mode; then
  exit 0
fi

if ! $no_music_mode ; then
  # recording should start first
  arecord -d 32 -f S16_LE -r 44100 -c 6 $recloc &
  # play music
  aplay -d 32 -r 44100 $musicloc
else
  arecord -d 32 -f S32_LE -r 44100 -c 6 $recloc
fi

echo done
