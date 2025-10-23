This repository provides the code for paper which has been submitted to ICASSP 2026. 
Active Sensing for Beam Alignment in Backscatter Communications

This paper studies an active sensing structure to learn beamforming vectors in a monostatic BC scenario to maximize SINR of the BD signal.


The training of the LSTM is proveded in rnn_server_wall.py and 2b.py.

They correspond to the simplified AP with one beamformer and AP with two beamformers.

**channel_functions.py** provides functions generating channels, beam_pattern and BD locations

**dnn.py** is tend for a benchmark where beamformers are generated through a DNN rather than RNN

**iter_gen_eig.py** provides functions of iterative optimization

**manifold_optimization.py**, **opti_transpose.py** give functions of manifold optimization for Raycian quotient for hermitian and transpose cases

**parse_args.py** gives command line tunnable parameters


**test_models_with_exact_structure.py** run a trained model to generate beamforming pattern


