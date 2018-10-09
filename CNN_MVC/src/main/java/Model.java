import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Random;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.ResNet50;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class Model {
	
	private static final int SEED = 123;
	final int MINI_BATCH_SIZE = 5;

	static int CHANNELS = 3;
	static int IMG_WIDTH = 400;
	static int IMG_HEIGHT = 300;

	static int N_LABELS = 3;
	
	int numEpochs = 50;
	int transformedDataEpochs = 5;
	
	public ParentPathLabelGenerator labelGenerator;
	
	private ZooModel<?> zooModel;
	private DataSetIterator citoTrainDataSet;
	private static DataSetIterator citoTestDataSet;
	
	private InputSplit trainData;
	private InputSplit testData;
	private ImageRecordReader trainReader;
	private ImageRecordReader testReader;
	

	public void openDataSet() throws IOException {
		
		labelGenerator = new ParentPathLabelGenerator();
		File citoTrainRootDir = Paths.get("/Users/Lily/workspace/CNN_MVC/citologia/train").toFile();
		File citoTestRootDir = Paths.get("/Users/Lily/workspace/CNN_MVC/citologia/test").toFile();
		this.trainData = new FileSplit(citoTrainRootDir, BaseImageLoader.ALLOWED_FORMATS, new Random());
		this.testData = new FileSplit(citoTestRootDir, BaseImageLoader.ALLOWED_FORMATS, new Random());
		this.trainReader = new ImageRecordReader(IMG_WIDTH, IMG_HEIGHT, CHANNELS, labelGenerator);
		this.testReader = new ImageRecordReader(IMG_WIDTH, IMG_HEIGHT, CHANNELS, labelGenerator);
		System.out.println("initializing");
		trainReader.initialize(trainData);
		testReader.initialize(testData);
		System.out.println(trainReader.getLabels());
		System.out.println(testReader.getLabels());
		this.citoTrainDataSet = new RecordReaderDataSetIterator(trainReader, MINI_BATCH_SIZE, 1, N_LABELS);
		citoTestDataSet = new RecordReaderDataSetIterator(testReader, MINI_BATCH_SIZE, 1, N_LABELS);
		this.zooModel = new ResNet50(N_LABELS, SEED, 1);
		
	}
	
	public static int getSeed() {
		return SEED;
	}

	public void setZooModel(ZooModel<?> zooModel) {
		this.zooModel = zooModel;
	}
	
	public ZooModel<?> getZooModel() {
		return zooModel;
	}

	public DataSetIterator getCitoTrainDataSet() {
		return citoTrainDataSet;
	}

	public void setCitoTrainDataSet(DataSetIterator citoTrainDataSet) {
		this.citoTrainDataSet = citoTrainDataSet;
	}

	public static DataSetIterator getCitoTestDataSet() {
		return citoTestDataSet;
	}

	public void setCitoTestDataSet(DataSetIterator citoTestDataSet) {
		Model.citoTestDataSet = citoTestDataSet;
	}

	public InputSplit getTrainData() {
		return trainData;
	}

	public void setTrainData(InputSplit trainData) {
		this.trainData = trainData;
	}

	public InputSplit getTestData() {
		return testData;
	}

	public void setTestData(InputSplit testData) {
		this.testData = testData;
	}

	public ImageRecordReader getTrainReader() {
		return trainReader;
	}

	public void setTrainReader(ImageRecordReader trainReader) {
		this.trainReader = trainReader;
	}

	public ImageRecordReader getTestReader() {
		return testReader;
	}

	public void setTestReader(ImageRecordReader testReader) {
		this.testReader = testReader;
	}
	
}
