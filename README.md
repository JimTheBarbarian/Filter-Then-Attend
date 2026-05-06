This is the Github repo for Filter-Then-Attend: Improving Attention-based Time Series Forecasting with Spectral Filtering, recently accepted as an oral presentation at ICASSP 2026. 

Some notes about getting value out of this repo:
- The actual spectral filtering block can be found in spectcaster.py. For the other models that we use spectral filters in this paper, I just import the block in.
- The exp.py is tailored for distributed training. It should work fine with a single GPU, but I cannot guarantee that at the moment.
- I have added some additional toggles to aid in experimentation. There is:
- Exclude_experiment/exclude_channels. This can be used to only train/test on a select number of features from your dataset. I used it early in our work to determine on which parts of the datasets our models were struggling.
- filter_warmup_epochs: This is something I haven't tested yet, but if you would like to train the transformer first for some number of epochs, and only then train the filter, you can use this setting.

