import paho.mqtt.client as mqtt
from generic_audio_functions import play_file, get_sound_file_length


def play_done_callback(client, userdata, message):
    global play_done
    play_done = True
    print("play_done")


def rec_done_callback(client, userdata, message):
    global rec_done
    rec_done = True
    print("rec_done")


def Input_parsing(dist, duration, direction=0, LoS=True, edist=0, edirection=0, location="0", top=True, test=False):
    args = ""
    if test:
        args = "{} -t".format(args)

    args = "{} --distance {}".format(args, dist)
    args = "{} --direction {}".format(args, direction)

    if not LoS:
        args = "{} --LoS".format(args)
        args = "{} --edistance {}".format(args, edist)
        args = "{} --edirection {}".format(args, edirection)

    if top:    
        args = "{} --top".format(args)

    args = "{} --location {}".format(args, location)
    args = "{} --duration {}".format(args, duration)

    return args


distance_cm = 50

music_location = '../AAC/sample_chirps/'
# music_names = ['baseline', 'baseline_fast', 'balanced', 'fast']
music_names = ['fast']

# Length of the music files (seconds)
durations = []
music_padding_s = 0.15
for music in music_names:
    d = get_sound_file_length(music_location + music + '.wav') + music_padding_s
    durations.append(d * 1.25)

rec_done = False
play_done = False

client = mqtt.Client()

client.connect("192.168.1.196")

client.subscribe("rec_done")
client.subscribe("play_done")

client.message_callback_add("rec_done", rec_done_callback)
client.message_callback_add("play_done", play_done_callback)

# loop
client.loop_start()

msg = Input_parsing(dist=distance_cm, duration=durations[0])
print("playrec settings \n{}\n".format(msg))

for i in range(len(music_names)):
    for _ in range(15):
        print(music_names[i])

        rec_done = False
        play_done = True

        # Construct message
        msg = Input_parsing(dist=distance_cm, duration=durations[0])

        tx_args = "{} --music {}".format(msg, music_names[i])
        client.publish("playrec", tx_args)

        music = '../AAC/sample_chirps/{}.wav'.format(music_names[i])
        play_file(music, padding_duration_ms=music_padding_s*1000, add_padding_to_end=True)

        while not (rec_done and play_done):
            pass

        print("")

print("done")
print("playrec settings \n{}\n".format(msg))
client.loop_stop()
