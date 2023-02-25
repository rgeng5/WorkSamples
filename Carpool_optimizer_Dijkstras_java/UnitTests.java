// --== CS400 File Header Information ==--
// Name: <Ruiqi Geng>
// Email: < rgeng5@wisc.edu>
// Notes to Grader: <This is my own implementation of unit tests.>

/*** JUnit imports ***/
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;
/*** JUnit imports end  ***/
import java.util.Hashtable;

public class UnitTests {

    protected GraphADT<String> _instance = null;
    
    @BeforeEach
    public void createInstane() {
        _instance = new CS400Graph<String>();
	_instance.insertVertex("a");
	_instance.insertVertex("b");
	_instance.insertVertex("c");
	_instance.insertEdge("a","b",2);
	_instance.insertEdge("b","a",2);
	_instance.insertEdge("a","c",4);
	_instance.insertEdge("c","a",4);
	_instance.insertEdge("b","c",5);
	_instance.insertEdge("c","b",5);
    }

    @Test
    public void testInsertVertex() {
        assertEquals(3, _instance.getVertexCount(),"Can not insert vertex.");
    }
    
    @Test
    public void testInsertEdge() {
        assertEquals(2, _instance.getWeight("a","b"),"Edge does not exist.");
    }

    @Test
    public void testRemoveVertex() {
	_instance.removeVertex("a");
        assertEquals(2, _instance.getEdgeCount(),"Can not remove vertex.");
    }

    @Test
    public void testShortestPath() {
        String result = _instance.shortestPath("a", "c").toString();
	assertEquals("[a, b, c]", result,"Shortest path found is wrong.");
    }

    @Test
    public void testShortestPathDistance() {
        int result2 = _instance.getPathCost("a", "c");
        assertEquals(6, result2,"Distance of shortest path found is wrong.");
    }



}
