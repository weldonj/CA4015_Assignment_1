## INTRODUCTION AND DESCRIPTION OF DATASET

Each of the datasets which we are analysing comprise 63 older people (between 65-88 years) and 90 younger people (between 18-34 years) who completed a computerised version of the Iowa Gambling Task [IGT]. This is a widely used cognitive task for assessing the decision-making of a human being. Each dataset contains the parameters, for each subject, of a reinforcement learning model which was applied to simulate the subject's actions or decision-making process while completing the IGT. We are analysing three different models which were utilised in Lili's paper, namely the:

- Outcome Representation Learning Model [ORL] [1]
- Prospect Valence Learning Model with Delta [PVL-Delta] [2]
- Values-Plus-Perserverance Model [VPP] [3]

The parameters of these models can be interpreted as several different underlying physcological processes such as learning from experience and sensitivity to rewards and punishments etc. which contribute to the human decision-making process [4].  In our analysis, we will use a variety of clustering techniques in order to segment the subject group into distinct clusters. Intuitively, these clusters will represent groups of subjects who make decisions in a similar manner, according to the cognitive processes captured by the parameters of the various models.