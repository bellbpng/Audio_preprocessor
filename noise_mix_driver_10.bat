set root=C:\Anaconda3
call %root%\Scripts\activate.bat %root%

call conda env list
call conda activate image
call cd E:\STT_Accuracy_PreProcessor

#for /L %%i in (0,1,9) do (call python Noise_Mix.py --pathi E:\SBS_contents\Vocal_Remover_out_mono\dataset\instruments_mono --pathn E:\SBS_contents\Vocal_Remover_out_mono\dataset\noises --input SBS_C000%%i_inst --noise SBS_C000%%i_noise --output SBS_C000%%i_mix)
for /L %%i in (10,1,99) do (call python Noise_Mix.py --pathi E:\SBS_contents\Vocal_Remover_out_mono\dataset\instruments_mono --pathn E:\SBS_contents\Vocal_Remover_out_mono\dataset\noises --input SBS_C00%%i_inst --noise SBS_C00%%i_noise --output SBS_C00%%i_mix)
#for /L %%i in (100,1,999) do (call python Noise_Mix.py --pathi E:\SBS_contents\Vocal_Remover_out_mono\dataset\instruments_mono --pathn E:\SBS_contents\Vocal_Remover_out_mono\dataset\noises --input SBS_C0%%i_inst --noise SBS_C0%%i_noise --output SBS_C0%%i_mix)
#for /L %%i in (1000,1,2998) do (call python Noise_Mix.py --pathi E:\SBS_contents\Vocal_Remover_out_mono\dataset\instruments_mono --pathn E:\SBS_contents\Vocal_Remover_out_mono\dataset\noises --input SBS_C%%i_inst --noise SBS_C%%i_noise --output SBS_C%%i_mix)
