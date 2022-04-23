import numpy as np
import wave
import json



def MS_Section_Signal_Filtering(path,file_name):
        input_Ori=path+"\\"+file_name+"_mono.wav";
        input_V=path+"\\"+file_name+"_Voice_mono.wav";
        input_I=path+"\\"+file_name+"_Instrument_mono.wav";
        input_J=path+"\\"+file_name+"_MS_section.json"''
        print(input_V," : MS extraction started");
        
        #Voices process first
        spf_V = wave.open(input_V, "r");
        signal_V = spf_V.readframes(-1);
        signal_V = np.fromstring(signal_V, np.int16);
        fr = spf_V.getframerate();
        
        
        #######################
        #File Get Read
        spf_I = wave.open(input_I, "r");
        signal_I = spf_I.readframes(-1);
        signal_I = np.fromstring(signal_I, np.int16);
        
    
        spf = wave.open(input_Ori, "r");
        signal = spf.readframes(-1);
        signal = np.fromstring(signal, np.int16);
    
        nchannels = spf.getnchannels();
        sampwidth = spf.getsampwidth();
        nframes = spf.getnframes();
        comptype = spf.getcomptype();
        compname = spf.getcompname();
        
        
        
        with open(input_J, "r", encoding="utf8") as file: 
            contents = file.read(); # string 타입 
            json_data = json.loads(contents);
        
        only_V_bool = np.zeros(signal_V.size,np.int16);
        mixture_bool_high_speech = np.zeros(signal_V.size,np.int16);
        mixture_bool_high_song = np.zeros(signal_V.size,np.int16);
        mixture_bool_low = np.zeros(signal_V.size,np.int16);
        
        ms_scale = fr/1000
        
        for i in range(0,int(len(json_data["vonly_json"])),1):
            for j in range(int(json_data["vonly_json"][i]["start"]*ms_scale),int(json_data["vonly_json"][i]["end"]*ms_scale),1):
                only_V_bool[j]=1;
        
        for i in range(0,int(len(json_data["linst_json"])),1):
            for j in range(int(json_data["linst_json"][i]["start"]*ms_scale),int(json_data["linst_json"][i]["end"]*ms_scale),1):
                mixture_bool_high_speech[j]=1;
        
        for i in range(0,int(len(json_data["qinst_json"])),1):
            for j in range(int(json_data["qinst_json"][i]["start"]*ms_scale),int(json_data["qinst_json"][i]["end"]*ms_scale),1):
                mixture_bool_low[j]=1;
        
        for i in range(0,int(len(json_data["song_json"])),1):
            for j in range(int(json_data["song_json"][i]["start"]*ms_scale),int(json_data["song_json"][i]["end"]*ms_scale),1):
                mixture_bool_high_song[j]=1;
        
        
        
        
    
    
        output_file = path+"\\"+file_name+"_STT_Voice.wav";
        
        filtered_signal = signal_V * only_V_bool;
        filtered_signal = filtered_signal + signal_V * mixture_bool_high_speech+ np.int16(0.5+(signal_I * mixture_bool_high_speech)*0.3);
        filtered_signal = filtered_signal + signal_V * mixture_bool_high_song;
        filtered_signal = filtered_signal + signal * mixture_bool_low;
        
        
        
        wav_file = wave.open(output_file, "w");
        wav_file.setparams((nchannels, sampwidth, fr, nframes, comptype, compname));
        wav_file.writeframes(filtered_signal);
        wav_file.close();
        
        
        
        output_file = path+"\\"+file_name+"_STT_Noise.wav";
        noise_signal = signal - filtered_signal;
        
        wav_file = wave.open(output_file, "w");
        wav_file.setparams((nchannels, sampwidth, fr, nframes, comptype, compname));
        wav_file.writeframes(noise_signal);
        wav_file.close();
        
        
        ########################################################
        
        spf_V.close()
        spf_I.close()
        spf.close()
