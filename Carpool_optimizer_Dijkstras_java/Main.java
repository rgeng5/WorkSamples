// --== CS400 File Header Information ==--
// Name: <Ruiqi Geng>
// Email: <rgeng5@wisc.edu>
// Notes to Grader: <This is my own implementation of back-end and front-end applications.>


import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;

public class Main {
  
    public static void main(String[] args) {
	CS400Graph g = new CS400Graph();
	
	//Back-end portion
	/* Read in Data.txt*/
	try {
		File myObj = new File("Data.txt");
		Scanner myReader = new Scanner(myObj);
      		String data = myReader.nextLine();
		String[] vertexName = data.split(" "); //split line by space
      	
		for (int i=0; i<8; i++) {
			g.insertVertex(vertexName[i]);
		} // insert vertices according to the first line of Data.txt
		
		int count_weights=0;
		while (myReader.hasNextLine()) {
        		String data2 = myReader.nextLine();
			String[] weights = data2.split(" ");
      			for (int i=0; i<8; i++) {
                        	g.insertEdge(vertexName[count_weights], vertexName[i], Integer.parseInt(weights[i]));
                	} //insert edges according to the provided adjacency matrix
			count_weights++;
		}
      		myReader.close();
    	} catch (FileNotFoundException e) {
      		System.out.println("An error occurred.");
      		e.printStackTrace();
	}
	
	//Demo: specify start and end locations, average speed, and maximum tolerable extra time
	String startLoc = "driver7";
	String endLoc = "driver8";	
	int speed = 65;
	int tolerance = 50;
	
	int distStraight=g.getWeight(startLoc,endLoc);	
	int distCarpool=g.dijkstrasShortestPath(startLoc,endLoc).distance;
 
	double timeStraight = distStraight / speed; //time spent driving straight from start to end
	double timeCarpool = distCarpool / speed; //time spent carpooling
	double t = (timeCarpool / timeStraight - 1)*100;

	String result = g.shortestPath(startLoc, endLoc).toString();
	System.out.println("From "+startLoc+" to "+endLoc+", carpool pickup order is: " + result+", for an extra time of "+t+"%, at a default speed of 65 miles/hour.");
	
	//Check if tolerance is exceeded
	if (t > tolerance ) {
		System.out.println("From "+startLoc+" to "+endLoc+", carpool exceeds maximum tolerable extra time of "+tolerance+"%.");
	}
     	
	//Front-end portion
	//Check if front end is needed
	Scanner myObj0 = new Scanner(System.in);
	System.out.println("Do you want to use user input: (y/n)");
	String user = myObj0.nextLine();
 	if (user.equalsIgnoreCase("n")) {
	} else if (!user.equalsIgnoreCase("y")) {
		throw new IllegalArgumentException("Invalid Input");
	} else {
		Scanner myObj = new Scanner(System.in);
		System.out.println("Enter driver number: (1-8)<Integer>");
		int startLoc0 = Integer.parseInt(myObj.nextLine());
		if (startLoc0 > 8 || startLoc0 < 0 ){
			throw new IllegalArgumentException ("Please input an integer of [1,8]"); 
		}
		
		Scanner myObj2 = new Scanner(System.in);
		System.out.println("Enter destination number: (1-8)<Integer>");
		int endLoc0 = Integer.parseInt(myObj2.nextLine());
		if (startLoc0 > 8 || startLoc0 < 0 ){
			throw new IllegalArgumentException ("Please input an integer of [1,8]"); 
		}

		Scanner myObj3 = new Scanner(System.in);
		System.out.println("Enter mileage from driver to destination: <Integer>");
		int distStraight2 = myObj3.nextInt();
		Scanner myObj4 = new Scanner(System.in);
		System.out.println("Enter driver speed [miles/hr]: (1-100)<Integer>)");
		int speed2 = myObj4.nextInt();
		Scanner myObj5 = new Scanner(System.in);
		System.out.println("Enter maximum percentage of extra time you are willing to take: <Integer>");
		int tolerance2 = myObj5.nextInt();
		
		String startLoc2 = "driver"+startLoc0;
		String endLoc2 = "driver"+endLoc0;

		g.insertVertex(startLoc2);
		g.insertVertex(endLoc2);
		g.insertEdge(startLoc2, endLoc2, distStraight2);	
		distCarpool=g.dijkstrasShortestPath(startLoc2,endLoc2).distance;
		timeStraight = distStraight / speed2; //time spent driving straight from start to end
		timeCarpool = distCarpool / speed2; //time spent carpooling
		t = (timeCarpool / timeStraight - 1)*100;

		result = g.shortestPath(startLoc2, endLoc2).toString();
		System.out.println("From "+startLoc2+" to "+endLoc2+", carpool pickup order is: " + result+", for an extra time of "+t+"%, at an input speed of "+speed2+" miles/hour.");

		//Check if tolerance is exceeded
		if (t > tolerance2 ) {
			System.out.println("From "+startLoc2+" to "+endLoc2+", carpool exceeds maximum tolerable extra time of "+tolerance2+"%.");
		}
	}
    }
} 


