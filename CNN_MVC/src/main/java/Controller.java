import java.io.IOException;
import java.util.Random;

import org.datavec.image.transform.CropImageTransform;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.ScaleImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.zoo.PretrainedType;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Controller {
	private static Logger log = LoggerFactory.getLogger(Controller.class);
	static ComputationGraph modelTransfer;

	public void learn() throws IOException {
		Model model = new Model();
		model.openDataSet();
		ImageTransform[] transforms = getTransforms();

		ComputationGraph initializedZooModel = (ComputationGraph) model.getZooModel().initPretrained(PretrainedType.IMAGENET);
		FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder().learningRate(0.0001)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.NESTEROVS)
				.seed(123).build();
		System.out.println(initializedZooModel.summary());
		Controller.modelTransfer = new TransferLearning.GraphBuilder(initializedZooModel)
				.fineTuneConfiguration(fineTuneConf).setFeatureExtractor("flatten_3")
				.removeVertexKeepConnections("fc1000")
				.addLayer("fc1000",
						new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nIn(2048)
								.nOut(Model.N_LABELS).weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX).build(),
						"flatten_3")
				.build();
		modelTransfer.setListeners(new ScoreIterationListener(model.MINI_BATCH_SIZE));
		System.out.println(modelTransfer.summary());
		modelTransfer.setListeners(new ScoreIterationListener(model.MINI_BATCH_SIZE));
		log.info("Train model....");

		log.info("Training with original data");
		for (int i = 0; i < model.numEpochs; i++) {
			log.info("Epoch " + i);
			modelTransfer.fit(model.getCitoTrainDataSet());
		}

		log.info("Training with transformed data");
		for (int i = 0; i < model.transformedDataEpochs; i++) {
			log.info("Epoch " + i + " (transformed data)");
			for (int j = 0; j < transforms.length; j++) {
				ImageTransform imageTransform = transforms[j];
				log.info("Epoch " + i + " (transform " + imageTransform + ")");
				model.getTrainReader().initialize(model.getTrainData(), imageTransform);
				model.setCitoTrainDataSet(new RecordReaderDataSetIterator(model.getTrainReader(), model.MINI_BATCH_SIZE, 1, Model.N_LABELS));
				modelTransfer.fit(model.getCitoTrainDataSet());
			}
		}
	}

	private static ImageTransform[] getTransforms() {
		ImageTransform randCrop = new CropImageTransform(new Random(), 10);
		ImageTransform warpTransform = new WarpImageTransform(new Random(), 42);
		ImageTransform flip = new FlipImageTransform(new Random());
		ImageTransform scale = new ScaleImageTransform(new Random(), 1);
		return new ImageTransform[] { randCrop, warpTransform, flip, scale };
	}
}
