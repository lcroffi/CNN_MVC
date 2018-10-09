import java.io.IOException;

public class Inicio {

	public static void main(String[] args) throws IOException {
		Controller ctrl = new Controller();
		ctrl.learn();
		View v = new View();
		v.showResults();
	}
}
