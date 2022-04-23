import argparse

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt 
import wave


class STT_Voice_Extractor(object):

    def __init__(self):
        self.result = 0

    def make_signal_plot(file_name,signal,output_plot):
    
        plt.figure(figsize=(24,4));
        plt.title(file_name + " Signal Wave...");
        plt.xlabel(str(signal.size)+' samples ');
        plt.ylabel('Signal');
        plt.axis([0,signal.size, -32768, 32767]);
        plt.plot(signal);
        plt.savefig(output_plot);

    def Voice_Audio_PostProcessor(path,file_name,th_type):
        input_Ori=path+"\\"+file_name+"_mono.wav";
        input_V=path+"\\"+file_name+"_Vocals_mono.wav";
        input_I=path+"\\"+file_name+"_Instruments_mono.wav";
        print(input_V," : MS extraction started");

        #Voices process first
        spf_V = wave.open(input_V, "r");
        signal_V = spf_V.readframes(-1);
        signal_V = np.fromstring(signal_V, np.int16);
        fr = spf_V.getframerate();
        print("Vocal Extraction");
        #pos_signal array will have positive values of signal
        pos_signal_V=np.array(signal_V,np.int16);
        #Negative value remover
        for i in range(0,signal_V.size,1):
            value = signal_V[i];
            if (value<0):pos_signal_V[i]=0;
    
        #TI : Time Interval setting 0.1 second
        TI_size = int(signal_V.size/(fr/10))+1;
        #TI_pos_signal array will have 0.1s summation of pos_signal
        TI_pos_signal_V=np.zeros(TI_size,np.uint32);
    
        #initialize 
        k=0;index=0;
        pos_v=0;
        #Summation of sampling frequency/10 samples
        for i in range(0,signal_V.size,1):
            if (i==signal_V.size-1):
                pos_v=pos_v+pos_signal_V[i];
                TI_pos_signal_V[index]=pos_v;
            elif (k<(fr/10)):
                k=k+1;
                pos_v=pos_v+pos_signal_V[i];
            else:
                k=0;
                TI_pos_signal_V[index]=pos_v;
                pos_v=0;
                index=index+1;
    
        #This array will be used by filter
        TI_pos_bool_V=np.zeros(TI_size,np.uint32);
        #TI_mean calculation to use threshold
        TI_th_V = 500000;
        if (th_type==0):
            TI_th_V = np.mean(TI_pos_signal_V)*0.5;
    
    
        #signal mean value using signal strength compare
        for i in range(0,TI_size,1):
            if (TI_pos_signal_V[i]<(TI_th_V)):
                TI_pos_bool_V[i]=0;
            else:
                TI_pos_bool_V[i]=1;
    
        c_index_V=np.array([]);
        #Filter extansion in 5 index distance
        for i in range(0,TI_size-5,1):
            if (TI_pos_bool_V[i+1]==1 or TI_pos_bool_V[i+2]==1 or TI_pos_bool_V[i+3]==1or TI_pos_bool_V[i+4]==1or TI_pos_bool_V[i+5]==1):
                c_index_V=np.append(c_index_V,[i],0);                
        for i in range(5,TI_size,1):
            if (TI_pos_bool_V[i-1]==1 or TI_pos_bool_V[i-2]==1 or TI_pos_bool_V[i-3]==1 or TI_pos_bool_V[i-4]==1 or TI_pos_bool_V[i-5]==1):
                c_index_V=np.append(c_index_V,[i],0);
        
        for i in range(0,c_index_V.size,1):
            TI_pos_bool_V[int(c_index_V[i])]=1;
    
        TI_pos_bool_V_large=np.zeros(signal_V.size,np.int16)
        s_size=TI_pos_bool_V.size;
        l_size=signal_V.size;
        for i in range(0,l_size,1):
            TI_pos_bool_V_large[i]=TI_pos_bool_V[int((i/l_size)*s_size)];
        
        #######################
        #File Get Read(v)
        spf_I = wave.open(input_I, "r");
        signal_I = spf_I.readframes(-1);
        signal_I = np.fromstring(signal_I, np.int16);
        print("Instruments Extraction");
        #Positive value filtering(v)
        pos_signal_I=np.array(signal_I,np.int16);
        for i in range(0,signal_I.size,1):
            value = signal_I[i];
            if (value<0):pos_signal_I[i]=0;
        #Positive value 0.1s Summation(v)
        TI_pos_signal_I=np.zeros(TI_size,np.uint32);
        k=0;index=0;
        pos_v=0;
        for i in range(0,signal_I.size,1):
            if (i==signal_I.size-1):
                pos_v=pos_v+pos_signal_I[i];
                TI_pos_signal_I[index]=pos_v;
            elif (k<(fr/10)):
                k=k+1;
                pos_v=pos_v+pos_signal_I[i];
            else:
                k=0;
                TI_pos_signal_I[index]=pos_v;
                pos_v=0;
                index=index+1;
    
        #TI Time Interval(v)
        TI_pos_bool_I=np.zeros(TI_size,np.uint32);
        TI_th_I = 200000;
        if (th_type==0):
            TI_th_I = np.mean(TI_pos_signal_I)*0.3;
    
    
        for i in range(0,TI_size,1):
            if (TI_pos_signal_I[i]<(TI_th_I)):
                TI_pos_bool_I[i]=0;
            else:
                TI_pos_bool_I[i]=1;
    
        #####I_filtering(v)
        f_index=np.array([]); #false index
        t_index=np.array([]); #true  index
        
        k=30;#number of counting
        for i in range(0,TI_size-k,k):
            num=0;
            for j in range(i,i+k,1):
                if(TI_pos_bool_I[j]==1):
                    num=num+1;
                if (num<(k/2)):
                    f_index=np.append(f_index,[i],0);
                else:
                    t_index=np.append(t_index,[i],0);    
    
        for i in range(0,f_index.size,1):
            for j in range(int(f_index[i]),int(f_index[i])+k,1):
                TI_pos_bool_I[j]=0;
        for i in range(0,t_index.size,1):
            for j in range(int(t_index[i]),int(t_index[i])+k,1):
                TI_pos_bool_I[j]=1;
        
        #####bool to original size(v)
        TI_pos_bool_I_large=np.zeros(signal_I.size,np.int16)
        for i in range(0,l_size,1):
            TI_pos_bool_I_large[i]=TI_pos_bool_I[int((i/l_size)*s_size)];
        
        #######################
        cov_filter = np.array(0.0476);
        for i in range(0,20,1):
            cov_filter = np.append(cov_filter,0.0476);
    
        cov_TI_pos_V = np.convolve(TI_pos_signal_V,cov_filter);
        cov_TI_pos_I = np.convolve(TI_pos_signal_I,cov_filter);
        TI_pos_V = cov_TI_pos_V[10:cov_TI_pos_V.size-10];
        TI_pos_I = cov_TI_pos_I[10:cov_TI_pos_I.size-10];
    
    
        TI_pos_I_high_V =np.zeros(TI_size,np.int16);
        TI_pos_I_low_V =np.zeros(TI_size,np.int16);
        for i in range(0,TI_size,1):
            posv=TI_pos_V[i];
            posi=TI_pos_I[i];
            if (posi>posv*0.5):
                TI_pos_I_high_V[i]=1;
            else:
                TI_pos_I_low_V[i]=1
    
        TI_pos_I_high_V_large=np.zeros(signal_I.size,np.int16);
        TI_pos_I_low_V_large=np.zeros(signal_I.size,np.int16);
    
        for i in range(0,l_size,1):
            TI_pos_I_high_V_large[i]=TI_pos_I_high_V[int((i/l_size)*s_size)];
            TI_pos_I_low_V_large[i]=TI_pos_I_low_V[int((i/l_size)*s_size)];
    
        #######################
        #frequency amplitude variance
        lpf=50;
        hpf=1000;
    
        #File Get & STFT spectrogram
        spf_f = wave.open(input_V);
        x, sr = librosa.load(input_V);
        var_list = np.zeros(TI_size);
    
        if (x.size/sr > 600):
            print("separation STFT");
            t_duration = int(sr*600);
            for d in range(0,x.size,t_duration):
                y = librosa.stft(x[d:d+t_duration], n_fft=sr, win_length = 128, hop_length=int(sr/10));
            
                magnitude = np.abs(y);
    
                freq_length=int(magnitude.size/magnitude[0].size);
                time_length=int(magnitude[0].size);
                samples = np.zeros((time_length,freq_length));
            
                for j in range(0,time_length,1):
                    for i in range(0,freq_length,1):
                        samples[j][i] = samples[j][i] + magnitude[i][j];
                    
                default_time = int(d/(sr/10))
                for i in range(0,time_length,1):
                    var_list[i+default_time]=np.var(samples[i][lpf:hpf]);
        
            remain_time = x.size%t_duration;
            #last_time = x.size()-remain_time;
            y = librosa.stft(x[x.size-remain_time:x.size], n_fft=sr, win_length = 128, hop_length=int(sr/10));
        
            magnitude = np.abs(y);
    
            freq_length=int(magnitude.size/magnitude[0].size);
            time_length=int(magnitude[0].size);
            samples = np.zeros((time_length,freq_length));
        
            for j in range(0,time_length,1):
                for i in range(0,freq_length,1):
                    samples[j][i] = samples[j][i] + magnitude[i][j];
        
            default_time = int((x.size-remain_time)/(sr/10));
            for i in range(0,time_length,1):
                var_list[i+default_time]=np.var(samples[i][lpf:hpf]);
        
        else:
            y = librosa.stft(x, n_fft=sr, win_length = 128, hop_length=int(sr/10));
    
            magnitude = np.abs(y);
            
            freq_length=int(magnitude.size/magnitude[0].size);
            time_length=int(magnitude[0].size);
            samples = np.zeros((time_length,freq_length));
    
            for j in range(0,time_length,1):
                for i in range(0,freq_length,1):
                    samples[j][i] = samples[j][i] + magnitude[i][j];
    
    
            for i in range(0,TI_size,1):
                var_list[i]=np.var(samples[i][lpf:hpf]);
    
    
        spf_f.close();
    
        ########################################################
    
    
    
        G_filter=np.array([0.0342500,0.0376634,0.0410047,0.0441984,0.0471667,0.0498335,0.0521272,0.0539840,0.0553506,0.0561871,0.0564688,0.0561871,0.0553506,0.0539839,0.0521272,0.0498335,0.0471667,0.0441984,0.0410047,0.0376634,0.0342500])

        filterd_var_list = np.convolve(var_list,G_filter)[10:var_list.size+10];
        Rescale_filterd_TI_pos_signal_V=TI_pos_V/18062808.75;
    
        song_bool_list = np.zeros(TI_size,np.int16);
        for i in range(0,TI_size,1):
            if(filterd_var_list[i]<Rescale_filterd_TI_pos_signal_V[i]):
                song_bool_list[i]=1;
    
        if (np.sum(song_bool_list[0:10])<8):
            for j in range(0,10,1):
                song_bool_list[j]=0;
        for i in range(10,TI_size,1):
            if (song_bool_list[i-1]==1 and song_bool_list[i]==0):
                if (np.sum(song_bool_list[i-10:i])<8):
                    for j in range(i-10,i,1):
                        song_bool_list[j]=0;
    
        song_bool_list_large=np.zeros(signal_V.size,np.int16);    
        speech_bool_list_large=np.zeros(signal_V.size,np.int16);    
    
        for i in range(0,l_size,1):
            song_bool_list_large[i]=song_bool_list[int((i/l_size)*s_size)];
            if(song_bool_list_large[i]==0):
                speech_bool_list_large[i]=1;
        #TI_pos_bool_V_large : 원본 timescale 목소리 bool
        #TI_pos_bool_I_large : 원본 timescale 악기소리 bool
        only_V_bool = np.zeros(signal_V.size,np.int16);
        only_I_bool = np.zeros(signal_I.size,np.int16);
        mixture_bool = TI_pos_bool_V_large * TI_pos_bool_I_large;
        for i in range(0,signal_V.size,1):
            V_bool = TI_pos_bool_V_large[i];
            I_bool = TI_pos_bool_I_large[i];
            if(V_bool==1 and I_bool==0):only_V_bool[i]=1;
            if(V_bool==0 and I_bool==1):only_I_bool[i]=1;
    
    
        spf = wave.open(input_Ori, "r"); # 원본파일
        signal = spf.readframes(-1);
        signal = np.fromstring(signal, np.int16);
    
        nchannels = spf.getnchannels();
        sampwidth = spf.getsampwidth();
        nframes = spf.getnframes();
        comptype = spf.getcomptype();
        compname = spf.getcompname();
    
    
    
        mixture_bool_high = mixture_bool * TI_pos_I_high_V_large;
    
        mixture_bool_high_song=mixture_bool_high*song_bool_list_large;
        mixture_bool_high_speech=mixture_bool_high*speech_bool_list_large;
    
        mixture_bool_low = mixture_bool * TI_pos_I_low_V_large;
    
    
        output_file = path+"\\"+file_name+"_STT_Voice.wav";
        filtered_signal = signal_V * only_V_bool;
        filtered_signal = filtered_signal + signal_V * mixture_bool_high_speech+ np.int16(0.5+(signal_I * mixture_bool_high_speech)*0.3);
        filtered_signal = filtered_signal + signal_V * mixture_bool_high_song;
        filtered_signal = filtered_signal + signal * mixture_bool_low;
    
        output_plot= path+"\\"+file_name+"_STT_Voice.png";#####
        STT_Voice_Extractor.make_signal_plot(file_name+"_STT_Voice",filtered_signal,output_plot);
        wav_file = wave.open(output_file, "w");
        wav_file.setparams((nchannels, sampwidth, fr, nframes, comptype, compname));
        wav_file.writeframes(filtered_signal);
        wav_file.close();
    
        output_file = path+"\\"+file_name+"_STT_Noise.wav";
        noise_signal = signal - filtered_signal;
        output_plot= path+"\\"+file_name+"_STT_Noise.png";#####
        STT_Voice_Extractor.make_signal_plot(file_name+"_STT_Noise",noise_signal,output_plot);
        wav_file = wave.open(output_file, "w");
        wav_file.setparams((nchannels, sampwidth, fr, nframes, comptype, compname));
        wav_file.writeframes(noise_signal);
        wav_file.close();
    

        ########################################################
    
        spf_V.close()
        spf_I.close()
        spf.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--path', '-P', type=str, default='.')
    p.add_argument('--input', '-i', required=True)
    p.add_argument('--th', '-t', type=int, default=0)   
        #if 0  the relative value for each content
        #else  the reference absolute value is used for threshold
    args = p.parse_args()
    path = args.path;
    file_name = args.input;
    th_type = args.th;
    STT_Voice_Extractor.Voice_Audio_PostProcessor(path,file_name,th_type);
    
if __name__ == '__main__':
    main()