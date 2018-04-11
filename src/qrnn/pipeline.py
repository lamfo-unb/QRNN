from toolz.curried import *

@curry
def pipeline(dataset, learners):
    return pipe(learners,
                reversed,
                reduce(comp))(dataset)
