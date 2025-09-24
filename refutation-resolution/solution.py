import argparse

def parse_clause_line(line):
    """strips whitespace, converts to lowercase, ingores comments
       splits the line using " v " as the delimiter and returns a frozenset of literals
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    line = line.lower()
    literals = [lit.strip() for lit in line.split(" v ") if lit.strip()]
    if not literals:
        return None
    return frozenset(literals)
    

def load_clauses_list(file_path):
    """reads file that contains a list of clauses
       Each line of the file contains a single clause in CNF format.
    """
    clauses = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            clause = parse_clause_line(line)
            if clause is not None:
                clauses.append(clause)
    return clauses



def load_user_command_file(file_path):
    """returns a list of tuples (clause, command)
    each line: a clause + whitespace + command (?,+,-)
    """
    commands = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip().lower()
            if not line or line.startswith("#"):
                continue
            command = line[-1]
            clause_str = line[:-2].strip()
            clause = parse_clause_line(clause_str)
            if clause is not None:
                commands.append((clause, command))      
    return commands


class Clause:
    def __init__(self, literals, clause_id, parents=None):
        self.literals = frozenset(literals)     # set of literals that make up the clause
        # frozenset - immutable and hashable, content won't change during the resolution procedure, allows use of *Clause* as keys in dictionaries or add them to sets
        self.clause_id = clause_id
        self.parents = sorted(parents) if parents else None # parents - a list of parent clause ids if derived, otherwise none(if the clause is a premise, i.e given in the input)
        # if the clause is derived, the filed will contain ids of 2 parent clauses, sorted
        
    def __eq__(self, other):
        # without id and parent, because 2 clauses with idnetical literasls are the same
        return isinstance(other, Clause) and self.literals == other.literals    # checks to see if we already have this clause
    
    def __hash__(self):
        #even if two clauses are “equal” by __eq__, if they have different ids, their hash values will differ.
        return hash((self.literals, self.clause_id))
    
    def __str__(self):
        lits = ' v '.join(sorted(self.literals)) if self.literals else 'NILL'
        parent_str = f" ({', '.join(str(parent) for parent in self.parents)})" if self.parents else ''
        return f"{self.clause_id}. {lits}{parent_str}"


def negate_lit(lit):
    return lit[1:] if lit.startswith("~") else "~" + lit
  
    
def check_tautology(literals):
    """a clause containing a literal and its negation is a tautology"""
    for lit in literals:
        if negate_lit(lit) in literals:
            return True
    return False 

    
def check_redundancy(clause, list_of_clauses):
    # a clause is redundaant if there is another clause whose literals are a subset, and length check to check for a proper subset
    for other in list_of_clauses:
        if other.clause_id != clause.clause_id and other.literals.issubset(clause.literals) and len(other.literals) < len(clause.literals):
            return True
    return False


def plResolve(c1, c2, new_clause_id):
    # for each literal in c1, if its negation is in c2, then produce a resolvent
    for lit in c1.literals:
        complement = negate_lit(lit)
        if complement in c2.literals:
            # new list takes the union of the literals from both clauses and removes both the literal and its complement
            new_lits = (c1.literals | c2.literals) - {lit, complement}
            if check_tautology(new_lits):
                continue
            # tuple (True -> resolution is successful, Clause(resolvants literals, provided new id, parents set to record the derivation))
            return True, Clause(new_lits, new_clause_id, parents=[c1.clause_id, c2.clause_id])
    return False, None


            
            
def resolution_strategy(premises, sos):
    # allow resolution steps only when at least one of the participating clauses comes from the negated goal clauses set (sos set)
    # premises -> "known" clauses, sos -> list of Clause object derived from the negation of the goal clause
    kb = premises.copy() # premises + newly derived clauses appended = knowledge base
    sos_set = sos.copy() # denial clauses and their descendants
    newly_derived_id = max(c.clause_id for c in kb) + 1
    resolvents = set()      #  to track pairs of clause IDs that have already been resolved
    
    while True:
        newly_derived = []
        # cleanup: remove redundancies and tautologies
        premises = [c for c in premises if not check_redundancy(c, premises) and not check_tautology(c.literals)]
        sos_set = [c for c in sos_set if not check_redundancy(c, sos_set) and not check_tautology(c.literals)]
        
        #sorting the clauses so that clauses that have literal come first, maybe it's quicker
        sos_set = sorted(sos_set, key=lambda c: len(c.literals))
        
        for c1 in premises:
            for c2 in sos_set:
                pair = (min(c1.clause_id, c2.clause_id), max(c1.clause_id, c2.clause_id))        # sorted pair of clause ids to avoid duplicate work
                if pair in resolvents:
                    continue
                resolvents.add(pair)
                
                success, res = plResolve(c1, c2, newly_derived_id)
                if not success:
                    continue        # no resolution applicable for the clauses
                
                if any(res.literals == existing.literals for existing in kb):    # avoid duplicates -> skip if an equivalent clasue already exists
                    continue
                if check_redundancy(res, kb):     # check redundancy in newly derived
                    continue
                
                newly_derived_id += 1
                newly_derived.append(res)
                
                if len(res.literals) == 0:      # contradiction, the resolvent is empty aka NIL, wuhuu
                    kb.append(res)
                    return True, res, kb
                
        if not newly_derived:       # nothing new was derived
            return False, None, kb
        # add newly derived clauses to the knowledge base and the set-of-support set
        for c in newly_derived:
            kb.append(c)
            sos_set.append(c)
        # add newly derived to premises for future resolution steps
        premises.extend(newly_derived)
        
def print_proof(nil_clause, clause_list):
    """backtracking derivation steps that led to () aka nil"""
    # mapping clause_id to Clause objects using a dictionary
    clause_dict = {c.clause_id: c for c in clause_list}
    derivations = set()
    def collect(c):
        if c not in derivations:
            derivations.add(c)
            if c.parents:
                for parent_id in c.parents:
                    collect(clause_dict[parent_id])
    collect(nil_clause)
    
    derived_list = sorted(list(derivations), key=lambda c: c.clause_id)
    for c in derived_list:
        print(str(c))
    print("============================")
    
def parse_clause_list_file(filepath, cooking=False):
    """the last clause is the goal clause. the goal is removed, negated(each literal seperately) and appended as individual unit clauses
    for cooking mode, returns a list of clause lists"""
    
    lines = []
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            l = line.strip().lower()
            if l and not l.startswith("#"):
                lines.append(l)
        if cooking:
            return [l.split(" v ") for l in lines]
        else:
            goal = lines.pop()          # last line is the goal clause
            premises = [l.split(" v ") for l in lines]
            no_premises = len(premises)
            neg_goal = []
            for lit in goal.split(" v "):
                neg_goal.append([negate_lit(lit)])
                
            all_clauses = premises + neg_goal
            clause_sets = [set(clause) for clause in all_clauses]
            clause_objects = []
            new_clause_id = 1
            for lit in clause_sets:
                clause_objects.append(Clause(lit, new_clause_id))
                new_clause_id += 1
            return clause_objects, clause_objects[no_premises:], goal.split(" v ")
        
def process_command(kb, cmd_file):
    
    def execute_command(kb_list, cmd):
        if len(cmd) < 2:
            return kb_list
        op = cmd[-1]
        clause_str = cmd[:-2].strip()
        lits = clause_str.split(" v ")
        if op == "?":
            num_orig = len(kb_list)
            negated = []
            for lit in lits:
                if lit.startswith("~"):
                    negated.append([lit[1:]])
                else:
                    negated.append(["~" + lit])
            all_clauses = kb_list.copy() + negated
            clause_sets = [set(c) for c in all_clauses]
            clauses = []
            clause_id = 1
            for s in clause_sets:
                clauses.append(Clause(s, clause_id))
                clause_id += 1
            
            sos = clauses[num_orig:]
            success, nil_clause, derived = resolution_strategy(clauses[:num_orig], sos)
            if success:
                if len(negated) == 1:
                    theorem = negate_lit(lits[0])
                else:
                    theorem = " v ".join(sorted([negate_lit(lit) for lit in lits]))
                print(f"[CONCLUSION]: {clause_str} is true")
            else:
                print(f"[CONCLUSION]: {clause_str} is unknown")
            return kb_list
        elif op == "+":
            if lits not in kb_list:
                kb_list.append(lits)
                print(f"Added {clause_str}")
            return kb_list
        elif op == "-":
            removed = False
            for c in kb_list:
                if c == lits:
                    kb_list.remove(c)
                    removed = True
                    print(f"Removed {clause_str}")
                    break
            if not removed:
                print("Clause not found; nothing to remove")
            return kb_list
        else:
            return kb_list
        
    kb_clauses = kb.copy()
    with open(cmd_file, "r", encoding="utf-8") as f:
        for line in f:
            cmd = line.strip().lower()
            if not cmd or cmd.startswith("#"):
                continue
            print("\nUser's command:", cmd)
            kb_clauses = execute_command(kb_clauses, cmd)
    return kb_clauses
            
def run_cooking_mode(clause_list_file, cmd_file):
    kb = parse_clause_list_file(clause_list_file, cooking=True)
    print("Constructed with knowledge: ")
    for c in kb:
        print(" v ".join(c))
    process_command(kb, cmd_file)
            
        
def run_resolution(clauses_list_file):
    premises, neg_goal, goal = parse_clause_list_file(clauses_list_file, cooking=False)
    success, nil_clause, all_clauses = resolution_strategy(premises, neg_goal)
    if success:
        print("=============================")
        print_proof(nil_clause, all_clauses)
        if len(neg_goal) == 1:
            theorem = negate_lit(next(iter(neg_goal[0].literals)))
        else:
            theorem = " v ".join(sorted([negate_lit(next(iter(c.literals))) for c in neg_goal]))
        print(f"[CONCLUSION]: {' v '.join(goal)} is true")
    else:
        print(f"[CONCLUSION]: {' v '.join(goal)} is unknown")
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=["resolution", "cooking"])
    parser.add_argument("list_of_clauses", type=str)
    parser.add_argument("commands_file", type=str, nargs="?", default="")
    args = parser.parse_args()
    
    if args.mode == "resolution":
        run_resolution(args.list_of_clauses)
    elif args.mode == "cooking":
        run_cooking_mode(args.list_of_clauses, args.commands_file)
        
if __name__ == "__main__":
    main()




