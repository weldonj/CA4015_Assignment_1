## Summary 

We are confident that the clustering work we have done on this dataset is quite comprehensive and that we have managed to maximise the quantity of useful information that can be found as a result of this clustering. We will quickly go through the decisions we made in the previous section and explain the motivations behind them.

Our initial discussions in our breakout room gave us a solid starting point. We took that time to just examine the data and try get a high level grasp of what it contained. We tried to get a feel for the assignment itself and what our final deliverable may contain and look like. This also involved looking into Jupyter Book and figuring out how it worked as we had never seen it before. We had a chance to scan through Lili's paper to get some actual background on what the data was describing. This also showed us which of the datasets was the best one to use based on Lili's descriptions (VPP).

From previous modules and also from some pairplot analysis we knew that standardizing the dataset would be a very useful first step to prepare it for the clustering work. We standardised and normalised the data and were left with a nicely prepared dataframe. We didn't want to just randomly choose a clustering algorithm and randomly choose an amount of clusters to use so instead we researched some methods that might let us know the optimal choices. 

We used the silhoutte score metric to determine what algorithm would suit best, and we initially used our own initiative to determine an appropriate cluster amount (young vs old). We realised that this was leaving a lot of useful information behind, however, and so used an elbow curve to get the optimal cluster amount once we realised that we could get more useful information by going deeper with more clusters. Once these were decided on, we were then able to run the actual clustering algorithms.

It was important to be able to visualise the clusters once we had found them and we know that visualisation beyond 3 dimensions is an issue. We did some principal component analysis to reduce the data to 3 dimensions, allowing some nice visualisations of the clusters. 

We then wanted to give some descriptions of what we believed the clusters represented so we did this one by one. 

We were then left with what we believe is a solid and comprehensive clustering assessment of the dataset, which was the overall aim of the assignment.



## Learning Outcomes

The assignment description itself was quite vague and left a lot of room for interpretation. This was initially somewhat daunting as we were unsure of what direction we should go in. However, once we downloaded the datasets and got our hands dirty it became clearer what kind of outcomes we should aim for. 

As a result of the ambiguity of the assignment, we were forced to approach it in a way we might be expected to approach other data science tasks in employment in the industry, where we could be given a dataset and told to simply get useful information from it. The lack of hand holding really meant we had to think outside the box somewhat and do our own research. 

Having completed the assignment we felt that we learned a lot more than we would have if the requirements were far more rigid and there was more hand holding. We certainly feel that this method provides the maximum amount of learning benefit that will be very useful to us throughout the rest of the year and into employment. 