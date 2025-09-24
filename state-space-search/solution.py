import argparse
from collections import deque
import heapq

class Node:
    def __init__(self, state, parent=None, cost=0.0, heuristic=0.0):
        self.state = state              
        self.parent = parent            
        self.cost = cost                
        self.heuristic = heuristic      
        
    def path(self):
        """Rekonstruira put od s0 do ovog čvora"""
        path = []
        current = self
        while current:
            path.append(current.state)
            current = current.parent
        return path[::-1]           
    
    def total(self):
        """Vraća zbroj g(n) + h(n)"""
        return self.cost + self.heuristic

    def __lt__(self, other):
        """Usporedba za heapq - prvo po ukupnoj cijeni, zatim leksikografski
         Definira način usporedbe čvorova kada se koriste u heapu. """
        if self.total() == other.total():
            return self.state < other.state
        return self.total() < other.total()


# inspired by cs50 introduction to AI
class QueueFrontier:
    def __init__(self):
        self.frontier = deque()
        self.states = set()     	    #skup stanja (set) radi brzog provjeravanja članstva

    def add(self, node):
        if node.state not in self.states:       #dodaje čvor u red ako stanje još nije dodano
            self.frontier.append(node)
            self.states.add(node.state)

    def contains_state(self, state):
        return state in self.states

    def empty(self):
        return len(self.frontier) == 0

    def remove(self):       #Uklanja i vraća čvor s početka reda. Također uklanja stanje iz skupa.
        """fifo princip"""
        if self.empty():
            raise Exception("empty frontier")
        node = self.frontier.popleft()
        self.states.remove(node.state)
        return node

class PriorityQueueFrontier:
    def __init__(self):
        self.frontier = []
        self.state_entries = {}     # rječnik za praćenje stanja koja su u fronti

    def add(self, node):
        """Dodaje ili ažurira čvor u prioritetnom redu ako postoji jeftiniji put do 
        Ako čvor s istim stanjem već postoji, a novi put je jeftiniji, ažurira se postojeći čvor; inače, dodaje novi čvor."""
        if node.state in self.state_entries:
            existing = self.state_entries[node.state]
            # ažuriraj postojeći čvor ako je novi put jeftiniji
            if node.cost < existing.cost:
                existing.cost = node.cost
                existing.parent = node.parent
                heapq.heapify(self.frontier)    # Obnovi heap nakon modifikacije
        else:
            # Dodaj novi čvor u heap i rječnik
            heapq.heappush(self.frontier, node)
            self.state_entries[node.state] = node

    def contains_state(self, state):
        return state in self.state_entries

    def empty(self):
        return len(self.frontier) == 0

    def remove(self):
        """Uklanja i vraća čvor s najmanjim cost-om iz prioritetnog reda i briše njegovo stanje iz rječnika"""
        if self.empty():
            raise Exception("empty frontier")
        node = heapq.heappop(self.frontier)
        del self.state_entries[node.state]      # Očisti zapis iz rječnika
        return node

def load_state_file(file_path):
    with open(file_path, encoding='utf-8') as f:
        lines = f.readlines()
    clean_lines = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):       # preskoci komentare i prazne linije
            clean_lines.append(line)

    initial = clean_lines[0]                # početno stanje (s0)
    goal = clean_lines[1].split()           # ciljna stanja (lista)

    transitions = {}                        #  prijelazi
    for line in clean_lines[2:]:
        state_info = line.split(':', 1)
        state = state_info[0].strip()
        transitions[state] = {}
        for token in state_info[1].strip().split():
            next_state, cost = token.split(',', 1)
            transitions[state][next_state.strip()] = float(cost.strip())
    return initial, goal, transitions

def load_heuristic(file_path):
    heuristics = {}
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(':', 1)
            state = parts[0].strip()
            h_value = parts[1].strip()
            heuristics[state] = float(h_value)
    return heuristics

class Algorithm:
    def __init__(self, initial, goals, transitions, heuristics=None):
        self.initial = initial          #s0
        self.goals = set(goals)        # Skup ciljnih stanja
        self.transitions = transitions  # Rječnik prijelaza
        self.heuristics = heuristics or {}  # Heurističke procjene
        self.explored = {}              # Rječnik istraženih stanja s najnižim cost-om
        self.solution_node = None       # čvor s rješenjem
        self.visited_count = 0          # Broj posjećenih stanja

    def is_goal(self, state):
        """Provjerava je li state ciljno stanje"""
        return state in self.goals

    def get_neighbors(self, state):
        """Vraća sortiranu listu susjednih stanja za zadano stanje (abecednim redom)"""
        return sorted(self.transitions.get(state, {}).items(), key=lambda x: x[0])

    def bfs(self):
        frontier = QueueFrontier()      # struktura podataka (red) u koju ćemo dodavati čvorove koji čekaju da budu prošireni
        frontier.add(Node(self.initial))
        self.explored = set()           # za bilježenje stanja koja su već obrađena. omogućuje izbjegavanje ponovnog obrađivanja istih stanja.
        self.visited_count = 0          # za statistiku i ispis rezultata

        while not frontier.empty():
            node = frontier.remove()
            
             # skip već istražena stanja
            if node.state in self.explored:
                continue
              
            # Označi stanje kao istraženo  
            self.explored.add(node.state)       #Ako čvor nije bio obrađen, označavamo njegovo stanje kao obrađeno tako što ga dodajemo u skup explored
            self.visited_count += 1

            # Provjeri je li pronađeno ciljno stanje
            if self.is_goal(node.state):
                self.solution_node = node
                return True

            # expand -> dodaj sve susjede u frontier
            for next_state, cost in self.get_neighbors(node.state):         # Petlja se vrti kroz sve susjedne (moguće) prijelaze iz trenutnog stanja
                if not frontier.contains_state(next_state) and next_state not in self.explored:
                    child = Node(next_state, node, node.cost + cost)    # Ako susjed nije obrađen, stvara se novi čvor (child) za to stanje
                    frontier.add(child)

        return False        # cilj nije pronađen

    """Definira metodu za pretragu s jednolikom cijenom unutar klase. Ova metoda pronalazi put s najmanjim ukupnim troškom (g(n)) """
    def ucs(self):
        frontier = PriorityQueueFrontier()
        frontier.add(Node(self.initial, cost=0.0))
        self.explored = {}          # rječnik explored koji će za svako obrađeno stanje spremati najmanji trošak kojim je do njega došlo. (Ključ je stanje, a vrijednost je trošak do tog stanja.)
        self.visited_count = 0

        while not frontier.empty():
            node = frontier.remove()
           
          # Preskoči stanja s većim cost-om od prethodng 
            if node.state in self.explored:
                if self.explored[node.state] <= node.cost:
                    continue
                self.explored[node.state] = node.cost
                self.visited_count += 1

            if self.is_goal(node.state):
                self.solution_node = node
                return True

            # Proširi čvor i ažuriraj frontier
            for next_state, cost in self.get_neighbors(node.state):     #Izračunava se novi kumulativni trošak do susjednog stanja dodavanjem troška prijelaza (cost) na trošak trenutnog čvora (node.cost)
                new_cost = node.cost + cost
                if next_state in self.explored and self.explored[next_state] <= new_cost:       
                    continue
                    
                child = Node(next_state, node, new_cost)
                frontier.add(child)

        return False

    def astar(self):
        """kombinira stvarni trošak puta (g(n)) i heurističku procjenu (h(n)) kako bi uvijek proširio čvor s najmanjim zbrojem"""
        frontier = PriorityQueueFrontier()
        # Inicijaliziraj s0 s heurističkom procjenom 0
        initial_node = Node(self.initial, cost=0.0)
        initial_node.heuristic = self.heuristics.get(self.initial, 0.0)
        frontier.add(initial_node)
        
        self.explored = {}
        self.visited_count = 0

        while not frontier.empty():
            node = frontier.remove()
            
            if node.state in self.explored:
                if self.explored[node.state] <= node.cost:
                    continue
                self.explored[node.state] = node.cost
                self.visited_count += 1
                

            if self.is_goal(node.state):
                self.solution_node = node
                return True

            # Proširenje čvora s ažuriranjem heuristike
            for next_state, cost in self.get_neighbors(node.state):
                new_cost = node.cost + cost
                heuristic = self.heuristics.get(next_state, 0.0)        #Dohvaća se heuristička vrijednost za susjedno stanje. Ako nije definirana, koristi se 0.0 kao zadana vrijednost.
                child = Node(next_state, node, new_cost, heuristic)
                
                if next_state in self.explored and self.explored[next_state] <= new_cost:
                    continue
                    
                frontier.add(child)

        return False

    def get_results(self):
        """Priprema rezultata nakon uspješnog pretraživanja"""
        if not self.solution_node:
            return None
            
        path = self.solution_node.path()
        return {
            'found': True,
            'path': path,
            'cost': self.solution_node.cost,
            'visited': self.visited_count
        }

class HeuristicCheck:
    def __init__(self, heuristics, goals, transitions, heuristic_file):
        self.heuristics = heuristics
        self.goals = goals
        self.transitions = transitions
        self.heuristic_file = heuristic_file        # Ime fajla za ispis

    def compute_optimal_cost(self, state):
        """Računa optimalni cost puta od zadanog stanja do cilja korištenjem ucs()"""
        solver = Algorithm(state, self.goals, self.transitions)     #Stvara se instanca klase Algorithm (koja implementira pretragu)
        found = solver.ucs()                                        #pokušati pronaći optimalni put do cilja.
        return solver.solution_node.cost if found else float('inf')

    def check_optimistic(self):
        """Provjerava je li (h(s) <= h*(s))"""
        print(f"# HEURISTIC-OPTIMISTIC {self.heuristic_file}")
        all_ok = True
        
        for state in sorted(self.heuristics.keys()):
            h = self.heuristics[state]
            h_star = self.compute_optimal_cost(state)
            
            if h > h_star:
                status = "[ERR]"
                all_ok = False
            else:
                status = "[OK]"
                
            print(f"[CONDITION]: {status} h({state}) <= h*: {h:.1f} <= {h_star:.1f}")
        
        conclusion = "optimistic" if all_ok else "not optimistic"
        print(f"[CONCLUSION]: Heuristic is {conclusion}.")

    def check_consistent(self):
        """Provjerava je li (h(s) <= h(t) + c(s,t))"""
        print(f"# HEURISTIC-CONSISTENT {self.heuristic_file}")
        all_ok = True
        
        for state in sorted(self.transitions.keys()):
            if state not in self.heuristics:            # Ako trenutno stanje nije u heuristikama, preskoči se (jer nemamo procjenu za njega)
                continue
                
            h_s = self.heuristics[state]
            
            for next_state, cost in sorted(self.transitions[state].items()):
                h_t = self.heuristics.get(next_state, 0.0)      # Dohvaća se heuristička vrijednost h(t) za susjedno stanje; ako nije definirana, koristi se 0.0.
                
                if h_s > h_t + cost:
                    status = "[ERR]"
                    all_ok = False
                else:
                    status = "[OK]"
                    
                print(f"[CONDITION]: {status} h({state}) <= h({next_state}) + c: {h_s:.1f} <= {h_t:.1f} + {cost:.1f}")
        
        conclusion = "consistent" if all_ok else "not consistent"
        print(f"[CONCLUSION]: Heuristic is {conclusion}.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', choices=['bfs', 'ucs', 'astar'])
    parser.add_argument('--ss', required=True)
    parser.add_argument('--h')
    parser.add_argument('--check-optimistic', action='store_true')
    parser.add_argument('--check-consistent', action='store_true')
    args = parser.parse_args()

    initial, goals, transitions = load_state_file(args.ss)
    heuristics = load_heuristic(args.h) if args.h else {}

    if args.alg:
        solver = Algorithm(initial, goals, transitions, heuristics)
        algorithm = args.alg.lower()
        
        if algorithm == 'bfs':
            found = solver.bfs()
        elif algorithm == 'ucs':
            found = solver.ucs()
        elif algorithm == 'astar':
            found = solver.astar()

        results = solver.get_results()
        print(f"# {algorithm.upper()}" + (f" {args.h}" if algorithm == 'astar' else ""))
        
        if not results:
            print("[FOUND_SOLUTION]: no")
        else:
            print("[FOUND_SOLUTION]: yes")
            print(f"[STATES_VISITED]: {results['visited']}")
            print(f"[PATH_LENGTH]: {len(results['path'])}")
            print(f"[TOTAL_COST]: {results['cost']:.1f}")
            print(f"[PATH]: {' => '.join(results['path'])}")

    if args.check_optimistic:
        checker = HeuristicCheck(heuristics, goals, transitions, args.h)
        checker.check_optimistic()
        
    if args.check_consistent:
        checker = HeuristicCheck(heuristics, goals, transitions, args.h)
        checker.check_consistent()

if __name__ == '__main__':
    main()