set root=C:\Anaconda3
call %root%\Scripts\activate.bat %root%

call conda env list
call conda activate image
call cd E:\STT_Accuracy_PreProcessor

#for /L %%i in (0,1,9) do ( call python Signal_extractor.py --pathi E:\SBS_contents\Vocal_Remover_out_mono --patho E:\SBS_contents\Original_wav_mono --input SBS_C000%%i )
#for /L %%i in (10,1,99) do ( call python Signal_extractor.py --pathi E:\SBS_contents\Vocal_Remover_out_mono --patho E:\SBS_contents\Original_wav_mono --input SBS_C00%%i )
for /L %%i in (100,1,999) do ( call python Signal_extractor.py --pathi E:\SBS_contents\Vocal_Remover_out_mono --patho E:\SBS_contents\Original_wav_mono --input SBS_C0%%i )
#for /L %%i in (1000,1,2998) do ( call python Signal_extractor.py --pathi E:\SBS_contents\Vocal_Remover_out_mono --patho E:\SBS_contents\Original_wav_mono --input SBS_C%%i )