# -*- coding: utf-8 -*-
"""
Created on Thu Jan 8 11:37:20 2019

@author: Thore
"""
import random
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt
plt.ioff() # Stops graphs from being displayed in terminal.
import os

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
def write_solution(line_to_write, cons=True):
    # Write to console
    if cons:
        print(line_to_write)

    # Write to file
    with open('Solution.txt', 'a') as file:
        file.write(line_to_write + '\n')
    file.close()

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#
# http://www.gilles-bertrand.com/2014/03/dijkstra-algorithm-python-example-source-code-shortest-path.html
#
def dijkstra(graph, src, dest, visited=None, distances=None, predecessors=None):
    """ 
    Calculates a shortest path tree routed in src
    """
    # Some bookkeeping issues - Basically, since lists are mutable objects, 
    # and keyword arguments are evaluated at function definition time, every 
    # time you call the function, you get the same default value.
    if visited is None:
        visited = []  
    if distances is None:
        distances={}
    if predecessors is None:
        predecessors={}
    
    # A few sanity checks
    if src not in graph:
        raise TypeError('The root of the shortest path tree cannot be found')
    if dest not in graph:
        raise TypeError('The target of the shortest path cannot be found')    
    
    # Ending condition
    if src == dest:
        # We build the shortest path and display it
        path=[]
        pred=dest
        while pred != None:
            path.append(pred)
            pred=predecessors.get(pred,None)
        
        path.reverse()
        cost = distances[dest]
        write_solution('New Aircraft Route: ' + str(path) + ' cost= ' + str(cost))

        with open("Search_Data.csv", "a") as file:
            writer = csv.writer(file)
            writer.writerow(path)
        file.close()

    else:     
        # if it is the initial  run, initializes the cost
        if not visited: 
            distances[src]=0
        # visit the neighbors
        for neighbor in graph[src] :
            if neighbor not in visited:
                new_distance = distances[src] + graph[src][neighbor]
                if new_distance < distances.get(neighbor,float('inf')):
                    distances[neighbor] = new_distance
                    predecessors[neighbor] = src
        # mark as visited
        visited.append(src)
        # now that all neighbors have been visited: recurse                         
        # select the non visited node with lowest distance 'x'
        # run Dijskstra with src='x'
        unvisited={}
        for k in graph:
            if k not in visited:
                unvisited[k] = distances.get(k,float('inf'))        
        x=min(unvisited, key=unvisited.get)
        dijkstra(graph,x,dest,visited,distances,predecessors)     
    
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
def calculate_cost(dict_of_neighbors, data_paths):
    # Calculate the cost on each route
    Cost = []
    for row in range(len(data_paths)):
        cost_per_route = 0 
        for col in range(len(data_paths.iloc[row])-1):
            if isinstance(data_paths.iloc[row][col+1], str):
                a = data_paths.iloc[row][col]
                b = data_paths.iloc[row][col+1]
                #print(a,' ',b, ' ', graph[a][b])
                cost_per_route += dict_of_neighbors[a][b]
        #print(cost_per_route)    
        Cost.append(cost_per_route)

    # Store the cost information within the Solution.txt file    
    write_solution('Total Cost = ' + str(sum(Cost)))
    write_solution('Total number of aircraft on taxiway: ' + str(len(Cost)))

    with open("Total_Cost.txt", 'a') as file:
        file.write(str(sum(Cost)) + '\n')
    file.close()
    
    with open("Total_Aircraft.txt", 'a') as file:
        file.write(str(len(Cost)) + '\n')
    file.close()
        
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
def duplicates_by_column(df):
    """
    Returns a list of dups by column for all time-events.
    """
    # Look for any duplicates within the passed dataframe
    column_names = list('abcdefghijklmnop')
    lst = []
    
    for i in range(len(column_names)):
        # Get the column name
        xname = column_names[i]
        
        # Look for any duplicates, excluding any NaN
        dup = df.duplicated(subset=xname, keep=False) & df[xname].notnull()
     
        # Append the sum of dup's for this column
        lst.append(dup.sum())

    return lst

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
def get_dups (dp, terminals, xnames):
    # Remove all NaN rows
    dp.dropna(how="all", inplace=True)
    
    # Drop the first 'X' columns (This allows us to look forward by 'X' time-frames)
    deeper = dp.drop(xnames, axis=1)

    # Look for any duplicates within the next time-frame in any column
    dup = deeper.iloc[:,0].duplicated(keep='first') & deeper.iloc[:,0].notnull()
    print('PRINTING DUP')
    print(dup)
    
    # If a duplicate has been detected, check each occurrence to see if it 
    # can be delayed (Remain at a terminal)
    for i in range(len(dup)):
        if dup[i] == True and dp.iloc[i, 0] in terminals:
            dup[i] = True # Only delay aircrafts at terminals
        else:
            dup[i] = False
            # Instead of making it False, it could divert the aircraft.
            # Make the route more expensive and run Dijkstra again.
            # Perhaps only necessary in larger airport with more runways.
    
    # Return an updated data_path and T/F-series for duplicates
    return dp, dup

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
def scan_forward_x_time_events(data_paths, terminals, count):
    # Looking forward 'X' time steps. This should ensure that only 1 aircraft
    # is in any one location at a time. If a future conflict is discovered,
    # add an additional delay (shift right) to its existing position. 
    column_names = list('abcdefghijklmnop')
    xnames = column_names[0:count]
    total_sum = 0
    
    # Print Scan event
    print('Scanning',count,'time events ahead by removing column(s): ',xnames)

    # Check for inital duplicates and return 'X' time-events forward    
    data_paths, duplicate = get_dups(data_paths, terminals, xnames)
       
    # While duplicates exist, move the row(s) containing these duplicates to 
    # the right by 1 time-event
    while duplicate.sum():
        print('Duplicates Detected: ', duplicate.sum())
        with open("Duplicates.txt", 'a') as file:
            file.write('{0:d}\n'.format(duplicate.sum()))
        file.close()
        
        total_sum += duplicate.sum()
        # This loop will process 1 or more detected duplicates
        for row in range(len(duplicate)):
            if duplicate[row] == True:
                print('>>> Processing row: ', row)
                for col in range(len(data_paths.iloc[row])-1, 0, -1):
                    data_paths.iloc[row, col] = data_paths.iloc[row, col-1]
                        
        # Check for initial duplicates and return 'X' time-events forward     
        data_paths, duplicate = get_dups(data_paths, terminals, xnames)     
    
    return data_paths, total_sum


#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
def conflictCheck_FCFS(dict_of_neighbors, terminals):
    # Read all of the current (active) paths
    data_paths = pd.read_csv("Search_Data.csv", header=None, names=list('abcdefghijklmnop'))

    # Drop any rows that are all NaN values
    data_paths.dropna(how="all", inplace=True)
    
    # Display the current map
    print('Current Map\n', data_paths)
    dup_sum = 0
            
    # Look for any duplicate paths/routes when looking forward by 'all' time-frames
    for i in range(1, len(data_paths.columns)):
        data_paths, xsum = scan_forward_x_time_events(data_paths, terminals, count = i)
        dup_sum += xsum

    # Save the (new/updated) dataframe - 
    data_paths.to_csv('Search_Data.csv', index=False, header=False)

    # Display the updated map
    print('Updated Map, with ' + str(dup_sum) + ' duplicates corrected.\n', data_paths)
    write_solution('Updated Map, with %s duplicates corrected ' %str(dup_sum), False)
    data_paths.to_csv('Solution.txt', index=True, header=False, mode='a', sep='\t')
  
    # Cost should be calculated after the routes have been updated.
    # Calculate and store the costs for each route within the data_paths dataframe
    calculate_cost(dict_of_neighbors, data_paths)    
    
    # Report on any duplicates within each column, regardless of 'X' time-events
    lst = duplicates_by_column(data_paths)
    str1 = ''.join(str(e) for e in lst)
    write_solution('Number of duplicates by column: ' + str1)
    if sum(lst) > 0:
        print('>>> Warning! There are duplicates in the schedule!\n\n')


#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
def plot_full_airport_map(map, counter):
    """
    Create a simple plot showing the airport matrix and all of the aircraft routes
    """
    # Read-in the most recent csv file and plot each of the routes    
    try:
        df = pd.read_csv('Search_Data.csv', header=None, names=list('abcdefghijklmnop'))
        f = plt.figure()
        for row in range(len(df)):
            X = []
            Y = []
            for col in range(len(df.iloc[row])):
                if isinstance(df.iloc[row][col], str):
                    A = df.iloc[row][col]
                    x, y = map[A]
                    x = x + row*5 # Slight offset to help with readability
                    y = y + row*5 # Slight offset to help with readability
                    X.append(x)
                    Y.append(y)
                    plt.plot(X, Y, linestyle='dashed', linewidth = 1, marker='o')
                    plt.title('Future Taxiway Motions at Time %s' %(counter), size = 14)

        plt.ylim(0, 800) 
        plt.xlim(0, 800)                     
        plt.grid()
        f.savefig('./FutureMotion_Data/Future Taxiway Motions %s.pdf' %(counter))

    except FileNotFoundError:
        print('Not available')    

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
def plot_current_airport_map(map, counter):
    """
    Plot just the current location of each aircraft 
    """
    # Read-in the most recent csv file and plot just the Current location for each aircraft    
    try:
        df = pd.read_csv('Search_Data.csv', header=None, names=list('abcdefghijklmnop'))
        f = plt.figure()
        for row in range(len(df)):
            if isinstance(df.iloc[row][0], str):
                A = df.iloc[row][0]
                x, y = map[A]
                x = x + row*0 # Can be used to create an offset to help with readability
                y = y + row*0 # Can be used to create an offset to help with readability
                plt.plot(x, y, marker='o', color = 'r')
                plt.title('Current Aircraft Positions at Time %s' %(counter), size = 14)


        plt.ylim(0, 800) 
        plt.xlim(0, 800) 
        plt.grid()
        f.savefig("./CurrentAirplane_Data/Current Aircraft Position %s.pdf" %(counter))

    except FileNotFoundError:
        print('Not available')
        
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------            
def advance_one_time_event():
    # Removes the [0] column in the DataFrame, moving everything forward
    # by 1 tick.
    df = pd.read_csv('Search_Data.csv', header=None, names=list('abcdefghijklmnop'))
    df.dropna(how="all", inplace=True)
    
    # Drop the first column and save the results    
    df = df.drop(df.columns[0], axis = 1)
    df.dropna(how="all", inplace=True) 
    df.to_csv('Search_Data.csv', index=False, header=False)

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------            
def open_gate(find):
    # Ensure the gate is not already populated before inserting another aircraft.
    # Returns True if the gate is open
    try:
        df = pd.read_csv('Search_Data.csv', header=None, names=list('abcdefghijklmnop'))
        df.dropna(how="all", inplace=True) 
        value = len(df[df['a'].str.contains(find)])
        if value == 0:
            return True
        else:
            return False

    except FileNotFoundError:
        # If the file doesn't exist, you can't have an aircraft at that gate
        return True

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------            
def all_terminals_full(terminals):
    # Check if all of the terminals are full (i.e., a new aircraft can't be inserted)
    # Returns True if all of the gates are full
    try:
        df = pd.read_csv('Search_Data.csv', header=None, names=list('abcdefghijklmnop'))
        df.dropna(how="all", inplace=True) 
        # Current location [0] of all aircrafts
        a = set(df['a'])

        # Are all of the terminals fulls? Returns True if every terminal is being used, else False
        return a.issuperset(terminals)

    except FileNotFoundError:
        # If the file doesn't exist, you can't have an aircraft at that gate
        return False    

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------      
def main():
    '''
    This program simulates an airport taxiway and calculates the shortest path
    from a randomly selected gate to the runway. 
    The simulation initially asks the user to pick 'Method 1: FCFS' or 'Method 2:
    Closest'. 
    
    Method 1 simulates a First-Come, First-Serve scenario, where the 
    aircraft are prioritized depending on when they entered the taxiway. The 
    earliest aircraft have priority.
    
    Method 2 simulates a scenario, where aircraft closer to their designated runway are
    prioritized over aircraft further from their designated runway. This method
    might be useful in airport configurations with multiple runways. In the current
    configuration, this method will give similar results to Method 1.
    
    Once the user has selected a method, the algorithm creates an aircraft and
    the shortest path is calculated using Dijkstra's algorithm. The algorithm
    tests whether there are any conflicts such that 2 or more aircraft would 
    be at the same location at the same time.
    
    Once the algorithm has confirmed no conflicts will arise, the simulation
    outputs the current layout of the aircraft on the taxiway and their future
    paths to the runway. The entire system is then advanced by one time-frame.
    
    The program returns to the beginning and creates a random variable
    based on the uniform distribution. If the variable falls within pre-set
    parameters another aircraft is generated and the algorithm runs again. If 
    the variable falls outside the parameters, no aircraft is generated and 
    the entire system is advanced by one time-frame.
    
    '''
    gate_cost = 1 # Cost of having an aircraft sitting at the gate
    delay_cost = 1 # Cost of having an aircraft delayed/stopped on the taxiway
    dict_of_neighbors = {
            'A': {'A': gate_cost, 'H': 1},
            'B': {'B': gate_cost, 'I': 1},
            'C': {'C': gate_cost, 'J': 1},
            'D': {'D': gate_cost, 'K': 1},
            'E': {'E': gate_cost, 'L': 1},
            'F': {'F': gate_cost, 'M': 1},
            'G': {'G': gate_cost, 'N': 1},
            'H': {'H': delay_cost, 'I': 1, 'O': 2},
            'I': {'I': delay_cost, 'H': 1, 'J': 1, 'P': 2},
            'J': {'J': delay_cost, 'I': 1, 'K': 1, 'Q': 2},
            'K': {'K': delay_cost, 'J': 1, 'L': 1, 'R': 2},
            'L': {'L': delay_cost, 'K': 1, 'M': 1, 'S': 2},
            'M': {'M': delay_cost, 'L': 1, 'N': 1, 'T': 2},
            'N': {'N': delay_cost, 'M': 1, 'U': 2},
            'O': {'O': delay_cost, 'X': 2, 'P': 1, 'H': 2},
            'P': {'P': delay_cost, 'O': 1, 'I': 2, 'Q': 1},
            'Q': {'Q': delay_cost, 'P': 1, 'J': 2, 'R': 1},
            'R': {'R': delay_cost, 'Q': 1, 'K': 2, 'S': 1},
            'S': {'S': delay_cost, 'R': 1, 'L': 2, 'T': 1},
            'T': {'T': delay_cost, 'S': 1, 'M': 2, 'U': 2},
            'U': {'U': delay_cost, 'T': 2, 'N': 2, 'V': 2},
            'V': {'V': delay_cost, 'U': 2, 'Z': 2},
            'X': {'X': delay_cost, 'O': 2},
            'Z': {'Z': delay_cost, 'V': 2}
            }
              
    # Complete listing of all of the Terminals
    terminals = set(['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    
    map = {
        'A': [100, 100],
        'B': [200, 100],
        'C': [300, 100],
        'D': [400, 100],
        'E': [500, 100],
        'F': [600, 100],
        'G': [700, 100],
        'H': [100, 200],
        'I': [200, 200],
        'J': [300, 200],
        'K': [400, 200],
        'L': [500, 200],
        'M': [600, 200],
        'N': [700, 200],
        'O': [100, 400],
        'P': [200, 400],
        'Q': [300, 400],
        'R': [400, 400],
        'S': [500, 400],
        'T': [600, 400],
        'U': [700, 400],
        'V': [900, 400],
        'X': [100, 600],
        'Z': [900, 600]
        }    

    # Function taken from keithweaver
    # https://gist.github.com/keithweaver/562d3caa8650eefe7f84fa074e9ca949
    # Creates a folder in the current working directory and prints all the files
    # to it.
    def createFolder(directory):
        try:
            if not os.path.isdir(directory):
                os.makedirs(directory)
        except OSError:
            print ('Error creating directory. ')     
    
    # Creates folders in the current directory called FutureMotion_Data and
    # CurrentAirplane_Data
    createFolder('./FutureMotion_Data/')
    createFolder('./CurrentAirplane_Data/')
    
    # Clear all existing files == Total_Cost.txt; Solution.txt; Search_Data.csv;
    # Duplicates.txt; Terminals.txt
    if os.path.exists('Total_Cost.txt'):
        os.remove('Total_Cost.txt')
    if os.path.exists('Solution.txt'):
        os.remove('Solution.txt')
    if os.path.exists('Search_Data.csv'):
        os.remove('Search_Data.csv')
    if os.path.exists('Total_Aircraft.txt'):
        os.remove('Total_Aircraft.txt')
    if os.path.exists('Duplicates.txt'):
        os.remove('Duplicates.txt')
    if os.path.exists('Terminals.txt'):
        os.remove('Terminals.txt')
        
    # Set the total number of loops/iterations, initialize the counter
    # and determine the probability limit for the uniform probability function.

    print('>>> Method: First-Come, First-Serve (FCFS)')
    prob = int(input('>>> Enter the probability of another aircraft being generated \n>>> by entering a value between 0 and 100: '))
    termination_condition = int(input('\n\n>>> Enter the number of loops until termination (must be above 0): '))
    
    start_time = time.time()
    prob_limit = 100 - prob
    
    # Write openning header            
    write_solution('This file contains the possible routes and the solution')
    write_solution('for a uniform probability distribution of %i %%' %prob_limit)
    write_solution('---------------------------------------------------------')
    
    # Counters for number of times all terminals are busy and the number of times
    # the randomly picked terminal for generating a new aircraft is busy. 
    busy_terminal = 0
    all_terms_busy = 0
    counter = 0
    while counter < termination_condition:
        """
        If the random number is below a certain limit and it is not the first 5 runs ->
        do not generate another aircraft. Only advance the existing aircraft 
        by one time interval. A smaller limit increases the chance of 
        generating an aircraft. A limit of 100 makes it impossible to 
        generate another aircraft after the initial 5.
        """ 
        # Uniform distribution to randomly choose whether a new aircraft is created.
        # Create a random integer between [0, 100] inclusively 
        prob = int(round(random.uniform(0, 100),0))
        
        # Pick a random terminal for the airplane to start.
        start_node = random.choice(list(terminals))
        
        if prob < prob_limit and counter > 5:
            write_solution('\nNo new aircraft due to probability. Advancing 1 timeframe without aircraft placement.\n')
            write_solution('Time: ' + str(counter+1))
            conflictCheck_FCFS(dict_of_neighbors, terminals) 

        elif open_gate(start_node):
            write_solution('\nInserting a new aircraft')
            write_solution('Time: ' + str(counter+1))
            dijkstra(dict_of_neighbors, start_node, 'X')
            conflictCheck_FCFS(dict_of_neighbors, terminals)             

        elif all_terminals_full(terminals):
            write_solution('\nAll terminals are full. Advancing 1 timeframe to make a terminal available.')
            write_solution('Time: ' + str(counter+1))
            all_terms_busy += 1

        else:
            print('\nThe randomly selected terminal is busy.')
            busy_terminal += 1
            continue # Restart loop

        # Common Functions
        counter += 1
        plot_full_airport_map(map, counter)
        plot_current_airport_map(map, counter)
        advance_one_time_event()
        
    print('\nNumber of times all terminals were busy: ',all_terms_busy)
    print('Number of times a random busy terminal is encountered: ',busy_terminal)
    with open("Terminals.txt", 'a') as file:
        file.write('Number of times all terminals were busy: {0:d}\n'.format(all_terms_busy))
        file.write('Number of times a busy terminal is encountered: {0:d}\n'.format(busy_terminal))
    file.close()
    
    end_time = time.time()
    print('\nTotal time of script execution (sec): ' + f'{(end_time-start_time):.2f}')

if __name__ == "__main__":
    main()
    