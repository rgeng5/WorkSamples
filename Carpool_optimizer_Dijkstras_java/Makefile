default: run

Main.class: Main.java CS400Graph.java GraphADT.java
	javac Main.java CS400Graph.java GraphADT.java 

run: Main.class 
	java Main
test: 
	javac -cp .:junit5.jar UnitTests.java -Xlint
	java -jar junit5.jar --class-path . --scan-classpath
clean:
	rm *.class
