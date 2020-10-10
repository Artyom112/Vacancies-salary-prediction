# Vacancies-salary-prediction
Prediction of expected salary based on vacancy description

To use trained model, import tensorflow and pandas libraries:
```
from tensorflow import keras
import pandas as pd
```
Then, download the trained model's weights from a public folder on Dropbox. In colab the code looks like this:
```
!wget -O rnn_conv.h5 https://www.dropbox.com/home?preview=rnn_conv.h5
```
Then, create the model using downloaded weights:
```
model = keras.models.load_model('rnn_nonv.h5')
```
Create panda's dataframe object containing all the data for which you would like to make predictions. Then, predict salaries using predict command. For example:
```
data_pred = pd.DataFrame({'Id': [123, 124], 'Title': ['senior worker', 'core java engineer london , big data'], 
                          'FullDescription': ['takes care of sick patients', 'develops applications'],
                          'LocationRaw': ['North West	', 'Surrey'],
                          'LocationNormalized': ['North West London	', 'Surrey'],
                          'ContractType': ['NaN', 'NaN'],
                          'ContractTime': ['permanent', 'permanent'],
                          'Other': ['Other', 'Other'],
                          'Category': ['Healthcare & Nursing Jobs', 'IT Jobs'],
                          'SalaryRaw': ['15931 - 17082 per annum', '55000 - 70000 per annum + Bens, Bonus (up to 20%), Equity'],
                          'SalaryNormalized': [16506, 62500	],
                          'SourceName': ['totaljobs.com', 'careers4a.com	']})
                          
log_salary = model.predict(make_batch(data_pred))
```


**Short description of the training process:**

The data contains description of various vacancies avaliable in the UK. The data attributes used by the model are either text data or categoriacal. Additionally, the distribution of salaries is right-tailed which means that it is desirable to apply log transformation to make the data resemble normal distribution.

![](/images/salaries_dist.png)

Then, data is tokenized and word counter is created. The number of word occurences has the following distribution.

![](/images/wordcounts_dist.png)

Then, all seldom words are deleted and each word is mapped to its index that depends on the number of occurences (from 0 and up).

We apply three different models to data contained in Title, FullDescription and categorical attributes. Then, three resulting vectors are concatenated and fed to a dense layer followed by a dense layer with a single neuron. The architecture is as follows:

![](/images/model.png)

Several variations of the model were tested, starting from conolutional NNs and ending with recurrent. Hovewer, the best architecture turned out to be the combination of convolutional network applied to text data in Title and recurrent network applied to FullDescription.


