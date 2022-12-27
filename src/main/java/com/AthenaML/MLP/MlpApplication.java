package com.AthenaML.MLP;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.FashionMnist;
import ai.djl.engine.Engine;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.*;
import ai.djl.nn.core.Linear;
import ai.djl.training.*;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.initializer.NormalInitializer;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.optimizer.Sgd;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.io.IOException;


@SpringBootApplication
public class MlpApplication {
	static int numInputs = 784;
	static int numOutputs = 10;
	static int hiddenUnits = 256;
	static final NDManager ndManager = NDManager.newBaseManager();

	static final NDArray w1 = ndManager.randomNormal(0,0.01f, new Shape(numInputs, hiddenUnits), DataType.FLOAT32);
	static final NDArray b1 = ndManager.ones(new Shape(hiddenUnits));

	static final NDArray w2 = ndManager.randomNormal(0,0.01f, new Shape(hiddenUnits, numOutputs), DataType.FLOAT32);
	static final NDArray b2 = ndManager.ones(new Shape(numOutputs));

	static int numEpochs = Integer.getInteger("MAX_EPOCH", 10);
	static float learningRate = 0.5f;
	static double[] trainLoss = new double[numEpochs];
	static double[] testAccuracy = new double[numEpochs];
	static double[] epochCount = new double[numEpochs];
	static double[] trainAccuracy = new double[numEpochs];

	static float epochLoss = 0f;
	static float accuracyVal = 0f;

	public static void main(String[] args) throws TranslateException, IOException {
		SpringApplication.run(MlpApplication.class, args);

		int batchSize = 256;

		FashionMnist trainData = FashionMnist
				.builder()
				.optUsage(Dataset.Usage.TRAIN)
				.setSampling(batchSize, true)
				.build();

		FashionMnist testData = FashionMnist
				.builder()
				.optUsage(Dataset.Usage.TEST)
				.setSampling(batchSize, true)
				.build();

		trainData.prepare();
		testData.prepare();






		NDManager ndManager = NDManager.newBaseManager();

		NDArray w1 = ndManager.randomNormal(0,0.01f, new Shape(numInputs, hiddenUnits), DataType.FLOAT32);
		NDArray b1 = ndManager.ones(new Shape(hiddenUnits));

		NDArray w2 = ndManager.randomNormal(0,0.01f, new Shape(hiddenUnits, numOutputs), DataType.FLOAT32);
		NDArray b2 = ndManager.ones(new Shape(numOutputs));

		NDList params = new NDList(w1, b1, w1, b2);

		for (NDArray param: params){
			param.setRequiresGradient(true);
		}

		SequentialBlock net = new SequentialBlock();
		net.add(Blocks.batchFlattenBlock(784));
		net.add(Linear.builder().setUnits(256).build());
		net.add(Activation::relu);
		net.add(Linear.builder().setUnits(10).build());
		net.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);


		Tracker lrt = Tracker.fixed(0.5f);
		Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();

		Loss loss = Loss.softmaxCrossEntropyLoss();

		DefaultTrainingConfig defaultTrainingConfig = new DefaultTrainingConfig(loss)
				.optOptimizer(sgd)
				.optDevices(Engine.getInstance().getDevices())
				.addEvaluator(new Accuracy())
				.addTrainingListeners(TrainingListener.Defaults.logging());

		try(Model model = Model.newInstance("mlp")){
			model.setBlock(net);

			try(Trainer trainer = model.newTrainer(defaultTrainingConfig)){
				trainer.initialize(new Shape(1,784));
				trainer.setMetrics(new Metrics());

				EasyTrain.fit(trainer,numEpochs, trainData, testData);

				Metrics metrics = trainer.getMetrics();

				//System.out.println(metrics);
			}
		}
	}
}
