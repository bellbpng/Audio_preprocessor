import argparse
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import wave
from pydub import AudioSegment

class Signal_extractor(object):

    def __init__(self):
        pass

    def make_plot(title, signal, sr, output_plot):
        
        plt.figure(figsize=(24,4));
        plt.title(title + " Signal Wave...");
        plt.xlabel('Time');
        plt.ylabel('Amplitude');
        librosa.display.waveplot(signal, sr)
        plt.savefig(output_plot);

    def Audio_PostProcessor(path, file_name):
        input_I = path + "\\" + file_name + "_Instruments_mono.wav"

        #Instrument Process
        print(input_I, ":Instruments post-processing started")
        signal_I, sr_i = librosa.load(input_I) # wave 파일 읽어오기

        #Extract each signal plot
        output_plot= path+"\\signal_plots\\"+file_name+"_Vocals.png"
        Signal_extractor.make_plot("Vocals", signal_V, sr_v, output_plot)

        output_plot= path+"\\signal_plots\\"+file_name+"_Instruments.png"
        Signal_extractor.make_plot("Instruments", signal_I, sr_i, output_plot)

    def Signal_PostProcessor(path, file_name):
        input_V = path + "\\" + file_name + "_Vocals_mono.wav"

        print("Vocals post-processing started")
        audio_signal, sr = librosa.load(input_V) # librosa.load(file): 음원의 파형 데이터, sampling rate 반환
        audio_signal = np.fromstring(audio_signal, np.float16) # 2 byte

        # Extracting meaningful data
        limit = 0.7 # 어떤 limit 값이 적당할지..?
        for idx, val in np.ndenumerate(audio_signal):
            if val > 0.7:
                audio_signal[idx] = 1
            else:
                audio_signal[idx] = 0

    def Instrument_PostProcessor(path_i, path_o, file_name, th_type):
        input_I = path_i + "\\" + file_name + "_Instruments_mono.wav"
        spf_I = wave.open(input_I, "r")
        signal_I = spf_I.readframes(-1) # 모든 프레임 읽어 들이기
        signal_I = np.fromstring(signal_I, np.int16)
        fr = spf_I.getframerate()
        l_size = signal_I.size
        
        input_origin = path_o + "\\" + file_name + "_mono.wav"
        spf_O = wave.open(input_origin, "r")
        signal_origin = spf_O.readframes(-1)
        signal_origin = np.fromstring(signal_origin, np.int16)


        #TI : Time Interval setting 0.1 second
        TI_size = int(signal_I.size/(fr/10))+1

        print("Instruments extraction")

        # Positive value filter
        pos_signal_I = np.array(signal_I, np.int16)

        for i in range(0, signal_I.size, 1):
            value = signal_I[i]
            if(value<0): pos_signal_I[i] = 0

        # Positive value 0.1s Summation
        TI_pos_signal_I = np.zeros(TI_size, np.int32)

        k=0
        index=0
        pos_v=0
        for i in range(0,signal_I.size,1):
            if (i==signal_I.size-1):
                pos_v=pos_v+pos_signal_I[i]
                TI_pos_signal_I[index]=pos_v
            elif (k<(fr/10)):
                k=k+1
                pos_v=pos_v+pos_signal_I[i]
            else: # k == fr/10 일때마다 TI_pos_signal_I 에 누적합(pos_v)을 더한다.
                k=0;
                TI_pos_signal_I[index]=pos_v
                pos_v=0
                index=index+1

        #TI Time Interval
        TI_pos_bool_I=np.zeros(TI_size,np.uint32)
        s_size=TI_pos_bool_I.size
        TI_th_I = 200000 # Threshold
        if (th_type==0):
            TI_th_I = np.mean(TI_pos_signal_I)*0.2 

        for i in range(0,TI_size,1):
            if (TI_pos_signal_I[i]<(TI_th_I)):
                TI_pos_bool_I[i]=0;
            else:
                TI_pos_bool_I[i]=1;

        #I_filtering
        f_index=np.array([]) #false index
        t_index=np.array([]) #true  index

        k=30 #number of counting
        for i in range(0,TI_size-k,k):
            num=0
            for j in range(i,i+k,1):
                if(TI_pos_bool_I[j]==1):
                    num=num+1
                if (num<(k/2)):
                    f_index=np.append(f_index,[i],0);
                else:
                    t_index=np.append(t_index,[i],0);    

        for i in range(0,f_index.size,1):
            for j in range(int(f_index[i]),int(f_index[i])+k,1):
                TI_pos_bool_I[j]=0
        for i in range(0,t_index.size,1):
            for j in range(int(t_index[i]),int(t_index[i])+k,1):
                TI_pos_bool_I[j]=1
        
        #####bool to original size
        TI_pos_bool_I_large=np.zeros(signal_I.size,np.int16)
        for i in range(0,l_size,1):
            TI_pos_bool_I_large[i]=TI_pos_bool_I[int((i/l_size)*s_size)];
        
        
        start_idx = 0
        end_idx = 0

        nchannels = spf_O.getnchannels()
        sampwidth = spf_O.getsampwidth()
        nframes = spf_O.getnframes()
        comptype = spf_O.getcomptype()
        compname = spf_O.getcompname()

        cnt = 0
        i = 0
        final_signal = np.array([], dtype=np.int16)
        while i < l_size-1:
            if TI_pos_bool_I_large[i]==0:
                cnt = cnt + 1
                start_idx = i
                i = i + 1
                while TI_pos_bool_I_large[i] == 0:
                    end_idx = i
                    if i >= l_size-1:
                        break
                    i = i + 1
            
                filtered_signal = signal_origin[start_idx:end_idx]
                final_signal = np.append(final_signal, filtered_signal, axis=0)
                
            else:
                i = i + 1

        if final_signal.size >= 190000:
            wav_file = wave.open(path_i+"\\dataset\\instruments_mono\\"+file_name+"_inst.wav", "w")
            wav_file.setparams((nchannels, sampwidth, fr, nframes, comptype, compname))
            wav_file.writeframes(final_signal)
            wav_file.close()

    def Noise_extractor(path_i, path_o, file_name, th_type):
        input_I = path_i + "\\" + file_name + "_Instruments_mono.wav"
        spf_I = wave.open(input_I, "r")
        signal_I = spf_I.readframes(-1) # 모든 프레임 읽어 들이기
        signal_I = np.fromstring(signal_I, np.int16)
        fr = spf_I.getframerate()
        l_size = signal_I.size
        
        input_origin = path_o + "\\" + file_name + "_mono.wav"
        spf_O = wave.open(input_origin, "r")
        signal_origin = spf_O.readframes(-1)
        signal_origin = np.fromstring(signal_origin, np.int16)


        #TI : Time Interval setting 0.1 second
        TI_size = int(signal_I.size/(fr/10))+1

        print("Instruments extraction")

        # Positive value filter
        pos_signal_I = np.array(signal_I, np.int16)

        for i in range(0, signal_I.size, 1):
            value = signal_I[i]
            if(value<0): pos_signal_I[i] = 0

        # Positive value 0.1s Summation
        TI_pos_signal_I = np.zeros(TI_size, np.int32)

        k=0
        index=0
        pos_v=0
        for i in range(0,signal_I.size,1):
            if (i==signal_I.size-1):
                pos_v=pos_v+pos_signal_I[i]
                TI_pos_signal_I[index]=pos_v
            elif (k<(fr/10)):
                k=k+1
                pos_v=pos_v+pos_signal_I[i]
            else: # k == fr/10 일때마다 TI_pos_signal_I 에 누적합(pos_v)을 더한다.
                k=0;
                TI_pos_signal_I[index]=pos_v
                pos_v=0
                index=index+1

        #TI Time Interval
        TI_pos_bool_I=np.zeros(TI_size,np.uint32)
        s_size=TI_pos_bool_I.size
        TI_th_I = 200000 # Threshold
        if (th_type==0):
            TI_th_I = np.mean(TI_pos_signal_I)*0.2 

        for i in range(0,TI_size,1):
            if (TI_pos_signal_I[i]<(TI_th_I)):
                TI_pos_bool_I[i]=0;
            else:
                TI_pos_bool_I[i]=1;

        #I_filtering
        f_index=np.array([]) #false index
        t_index=np.array([]) #true  index

        k=30 #number of counting
        for i in range(0,TI_size-k,k):
            num=0
            for j in range(i,i+k,1):
                if(TI_pos_bool_I[j]==1):
                    num=num+1
                if (num<(k/2)):
                    f_index=np.append(f_index,[i],0);
                else:
                    t_index=np.append(t_index,[i],0);    

        for i in range(0,f_index.size,1):
            for j in range(int(f_index[i]),int(f_index[i])+k,1):
                TI_pos_bool_I[j]=0
        for i in range(0,t_index.size,1):
            for j in range(int(t_index[i]),int(t_index[i])+k,1):
                TI_pos_bool_I[j]=1
        
        #####bool to original size
        TI_pos_bool_I_large=np.zeros(signal_I.size,np.int16)
        for i in range(0,l_size,1):
            TI_pos_bool_I_large[i]=TI_pos_bool_I[int((i/l_size)*s_size)];
        
        
        start_idx = 0
        end_idx = 0

        nchannels = spf_O.getnchannels()
        sampwidth = spf_O.getsampwidth()
        nframes = spf_O.getnframes()
        comptype = spf_O.getcomptype()
        compname = spf_O.getcompname()


        i = 0
        final_signal = np.array([], dtype=np.int16)
        while i < l_size-1:
            if TI_pos_bool_I_large[i]==1:
                start_idx = i
                i = i + 1
                while TI_pos_bool_I_large[i] == 1:
                    end_idx = i
                    if i >= l_size-1:
                        break
                    i = i + 1
            
                filtered_signal = signal_I[start_idx:end_idx]
                final_signal = np.append(final_signal, filtered_signal, axis=0)
                
            else:
                i = i + 1

        wav_file = wave.open(path_i+"\\dataset\\noises\\"+file_name+"_noise.wav", "w")
        wav_file.setparams((nchannels, sampwidth, fr, nframes, comptype, compname))
        wav_file.writeframes(final_signal)
        wav_file.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pathi', '-P', type=str, default='.')
    p.add_argument('--patho', '-po', type=str, default='.')
    p.add_argument('--input', '-i', required=True)
    p.add_argument('--th', '-t', type=int, default=0)

    args = p.parse_args()
    path_i = args.pathi
    path_origin = args.patho
    file_name = args.input
    th_type = args.th

    Signal_extractor.Instrument_PostProcessor(path_i, path_origin, file_name, th_type)
    #Signal_extractor.Noise_extractor(path_i, path_origin, file_name, th_type)

if __name__ == '__main__':
    main()

