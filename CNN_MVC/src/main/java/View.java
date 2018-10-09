import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.io.Files;

public class View {
	private static Logger log = LoggerFactory.getLogger(Controller.class);
	
	public void showResults() throws IOException {
		log.info("Evaluate model....");
		Evaluation eval = new Evaluation(Model.N_LABELS);
		while (Model.getCitoTestDataSet().hasNext()) {
			DataSet next = Model.getCitoTestDataSet().next();
			INDArray[] output = Controller.modelTransfer.output(next.getFeatureMatrix());
			for (int i = 0; i < output.length; i++) {
				eval.eval(next.getLabels(), output[i]);
			}
		}

		String stats = eval.stats();
		log.info(stats);
		log.info("****************Example finished********************");
		Files.write(stats.getBytes(), Paths.get("latest-output.txt").toFile());
		File file = new File("citologia-model.zip");
		ModelSerializer.writeModel(Controller.modelTransfer, file, true);
	}
}
