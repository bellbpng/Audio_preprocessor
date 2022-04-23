import argparse
import os

from pydub import AudioSegment
import numpy as np
import librosa
import wave

class wav_to_stereo(object):
    def __init__(self):
        self.result = 0

    def mono_input_to_stereo(path_i,path_noise,file_answer,file_noise,file_out):
        input_file = path_i+"\\"+file_answer+".wav";
        noise_file = path_noise+"\\"+file_noise+".wav";
        path = "E:\\SBS_contents\\Vocal_Remover_out_mono\\dataset";
        try:
            if not os.path.exists(path+"\\mixtures"):
                os.makedirs(path+"\\mixtures")
        except OSError:
            print ('Error: Creating directory. ' +  path+"\\mixtures");
        try:
            if not os.path.exists(path+"\\instruments"):
                os.makedirs(path+"\\instruments")
        except OSError:
            print ('Error: Creating directory. ' +  path+"\\instruments");
        
        #output_file = path+"\\"+file_type+"\\"+file_out+".wav";
        
        input_stereo = path+"\\instruments\\st_"+file_answer+".wav";
        sound = AudioSegment.from_wav(input_file);
        sound = sound.set_channels(2);
        sound.export(input_stereo, format="wav");
        
        #sound_n = AudioSegment.from_wav(noise_file);
        
        spf = wave.open(input_stereo, "r");
        signal = spf.readframes(-1);
        signal = np.fromstring(signal, np.int16);
    
        fr = spf.getframerate();
        nchannels = spf.getnchannels();
        sampwidth = spf.getsampwidth();
        nframes = spf.getnframes();
        comptype = spf.getcomptype();
        compname = spf.getcompname();
        
        spf_N = wave.open(noise_file, "r");
        signal_N = spf_N.readframes(-1);
        signal_N = np.fromstring(signal_N, np.int16);

        if (signal.size<signal_N.size*2):
            for i in range(0,signal.size,2):
                signal[i]=signal_N[int(i/2)];
                
        else:
            size_N=len(signal_N)*2;
            for i in range(0,signal.size,2):
                signal[i]=signal_N[int((i%size_N)/2)];
        
        temp_file = path+"\\mixtures\\temp_"+file_out+".wav";
        
        wav_file = wave.open(temp_file, "w");
        wav_file.setparams((nchannels, sampwidth, fr, nframes, comptype, compname));
        wav_file.writeframes(signal);
        wav_file.close();
        
        
        
        #sound.export(output_file, format="wav");
        
        output_stereo = path+"\\mixtures\\st_"+file_out+".wav";
        sound = AudioSegment.from_wav(temp_file);
        sound = sound.set_channels(1);
        sound = sound.set_channels(2);
        sound.export(output_stereo, format="wav");
        
        
        os.system("erase "+temp_file);


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pathi', '-P', type=str, default='./')
    p.add_argument('--pathn', '-p', type=str, default='./')
    p.add_argument('--input', '-i', required=True)
    p.add_argument('--noise', '-n', required=True)
    p.add_argument('--output', '-o', required=True)
    args = p.parse_args()
    
    path_i = args.pathi;
    path_noise = args.pathn;
    #file_name = args.input;
    
    file_answer = args.input;
    file_noise = args.noise;
    
    file_out = args.output;

    wav_to_stereo.mono_input_to_stereo(path_i,path_noise,file_answer,file_noise,file_out);
    
    
    
if __name__ == '__main__':
    main()