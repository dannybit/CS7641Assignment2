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
from helpers import initialize_instances,do_train
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm


ds_name = 'spam'
INPUT_LAYER = 57
HIDDEN_LAYER1 = 57
HIDDEN_LAYER2 = 57
OUTPUT_LAYER = 1
TRAINING_ITERATIONS = 5001
OUTFILE = 'output/{}/{}_NN_LOG.txt'


def run(P,mate,mutate):
    train = initialize_instances('{}_train.csv'.format(ds_name))
    val = initialize_instances('{}_val.csv'.format(ds_name))
    test = initialize_instances('{}_test.csv'.format(ds_name))

    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    dataset = DataSet(train)
    relu = RELU()
    rule = RPROPUpdateRule()
    classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER1, HIDDEN_LAYER2, OUTPUT_LAYER],relu)
    nnop = NeuralNetworkOptimizationProblem(dataset, classification_network, measure)
    trainer = StandardGeneticAlgorithm(P, mate, mutate, nnop)
    oa_name = "GA_{}_{}_{}".format(P,mate,mutate)
    do_train(trainer, classification_network, train, val, test, measure, TRAINING_ITERATIONS, OUTFILE.format(ds_name, oa_name))


run(50, 10, 10)
run(50, 30, 10)
run(50, 40, 10)
run(50, 50, 10)
run(100, 75, 10)
run(100, 100, 10)
run(100, 50, 20)