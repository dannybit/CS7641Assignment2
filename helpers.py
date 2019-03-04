import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "ABAGAIL/ABAGAIL.jar"))
from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem
from func.nn.backprop import RPROPUpdateRule, BatchBackPropagationTrainer
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import csv
from func.nn.activation import RELU
import time

def initialize_instances(file):
    instances = []
    with open(file, "r") as data:
        reader = csv.reader(data)
        for row in reader:
            instance = Instance([float(value) for value in row[:-1]])
            instance.setLabel(Instance(0 if float(row[-1]) == 0 else 1))
            instances.append(instance)
        return instances

# Adapted from:
# https://codereview.stackexchange.com/questions/36096/implementing-f1-score
# https://www.kaggle.com/hongweizhang/how-to-calculate-f1-score
# https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/
def f1_score(labels, predicted):
    get_count = lambda x: sum([1 for i in x if i is True])

    tp = get_count([predicted[i] == x and x == 1.0 for i, x in enumerate(labels)])
    tn = get_count([predicted[i] == x and x == 0.0 for i, x in enumerate(labels)])
    fp = get_count([predicted[i] == 1.0 and x == 0.0 for i, x in enumerate(labels)])
    fn = get_count([predicted[i] == 0.0 and x == 1.0 for i, x in enumerate(labels)])

    if tp == 0:
        return 0, 0, 0

    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)

    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        return precision, recall, 0.0
    return precision, recall, f1

def errorOnDataSet(network, ds, measure):
    N = len(ds)
    error = 0
    correct = 0
    incorrect = 0
    actuals = []
    predicteds = []
    for instance in ds:
        network.setInputValues(instance.getData())
        network.run()
        actual = instance.getLabel().getContinuous()
        predicted = network.getOutputValues().get(0)
        predicted = max(min(predicted, 1),0)
        predicteds.append(round(predicted))
        actuals.append(max(min(actual, 1), 0))
        if abs(predicted - actual) < 0.5:
            correct += 1
        else:
            incorrect += 1

        output = instance.getLabel()
        output_values = network.getOutputValues()
        example = Instance(output_values, Instance(output_values.get(0)))
        error += measure.value(output, example)
    MSE = error/float(N)
    acc = correct/float(correct+incorrect)
    precision, recall, f1 = f1_score(actuals, predicteds)
    return MSE,acc,f1

def do_train(trainer, network, train, val, test, measure, training_iterations, outfile):
    times = [0]
    for iteration in xrange(training_iterations):
        start = time.clock()
        trainer.train()
        elapsed = time.clock() - start
        times.append(times[-1]+elapsed)
        if iteration % 10 == 0:
            mse_train, acc_train, f1_train = errorOnDataSet(network, train, measure)
            mse_val, acc_val, f1_val = errorOnDataSet(network, val, measure)
            mse_test, acc_test, f1_test = errorOnDataSet(network, test, measure)
            txt = '{},{},{},{},{},{},{},{},{},{},{}\n'.format(iteration, mse_train, mse_val, mse_test, acc_train, acc_val,
                                                              acc_test, f1_train, f1_val, f1_test, times[-1])
            print(txt)
            with open(outfile, 'a+') as f:
                f.write(txt)