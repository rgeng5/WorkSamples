# Carpool optimizer

**Project Description**

When a person drives to a destination, they have the option to pick up someone else
along the way. This application allows users to figure out whether it is feasible to pick up
someone requesting a lift, and if so, the best order to pick up the riders. The data set contains
the distance from one rider's start and end points to another's (10 requests in total) and from
each point to the destination. The user inputs the maximum extra time they are willing to
spend to get to their destination due to carpooling. The app determines if there are eligible
riders and if so, each eligible rider is given an estimated time of arrival based on the
distances traversed and a default driving speed. Primary users of the app are people who
consider carpooling to events or work.



**Data Structure/Algorithm and Justification **

Dijkstra's Shortest Path Algorithm is used because it calculates the best traversal path
from a starting node to a destination on a weighted graph. The driving distances in question
is used as weights, and each rider’s start position and end position are inserted as 2
additional nodes. The output of the Dijkstra’s algorithm tells us which riders are eligible
for pickup, so that the total driving distance does not exceed user’s tolerance.


**Unit Tests **

1. Validate that invalid inputs (string, special characters) are handled correctly
2. Specific edge cases: a rider with the same start and end point as the driver should be
picked
3. Specific edge cases: a rider very far start and/or end point should not be picked
4. Verify the result of the Dijkstra’s algorithm is indeed the shortest path
5. Validate the estimated time of arrival for each eligible rider