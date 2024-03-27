# NFL-Injury-Predictions
The Natonal Football League is the premier American Football organization and one of the largest sports
leagues in the world. Unfortunately, its players experience injuries at an extremely high rate. This is a
growing criticism of the sport, and a detriment to both the potental growth of the league and the safety
of its players. Identifying the relationships that various game-related factors such as surface type,
temperature, and player position have on player injuries would not only be useful for identifying
injurious trends but also help to potentially mitigate the overall injury rate of the league.
Previous research is largely targeted towards either ligament-specific injuries or head trauma as a
measurement of injury, but research about general player injury- even minor, is sparse. Additionally,
there are very few studies that consider environmental factors as a catalyst for injury. We use several
years’ worth of nflfastR’s play-by-play data to aggregate a dataset that details every play that occurred
during the 2019 and 2023 seasons, then created our own injury identification variable using the
description of each play to determine if the resulting outcome of each play featured an injury (regardless
of its severity). We attempt to discover correlation between injury occurrence and the aforementioned
game variables using exploratory data analysis, and then utilized various predictive modeling techniques
such as: logistic regression, random forest, and naive bayes classification, to predict potential injury
occurrence for a given play.
Ultimately, we found the random forest model to be most effective at predicting injuries. It is largely
dependent on player position and an engineered feature called ’injury_frequency’. Aside from these
features, other models found variables related to downfield passing to be useful predictors of injury as
well. As the NFL becomes more and more passing focused, the implications of our findings as they relate
to potential injury mitiga.on are worth exploring.
An additional analysis, using naïve bayes, on which player positions were most susceptible to injury
showed that several groups had a higher incidence of injury as a result of a larger volume of usage. With
predictive capabilities, this could make a case for certain groups to rest and rotate more frequently
during a game. This could be an especially important research question moving forward as the game
continues to get faster and therefore, incorporate a higher volume of total plays.
