import networkx as nx
import matplotlib.pyplot as plt
from random import choices, sample


class Pandemic_Network:
    N = 10000
    k = 6
    r = 0.1
    p = 0.1
    d = 6
    number_of_initial_infected = 10
    infected = {
        "status": "infected",
        "days_with_disease": 1
    }
    recovered = {
        "status": "recovered",
    }
    G: nx.Graph
    cumulative_cases = []
    daily_cases = [0]

    def __init__(self):
        self.generate_pandemic_network(self.N, self.k, self.p)
        num_edges = self.G.number_of_edges()
        num_nodes = self.G.number_of_nodes()
        average_degree = num_edges / num_nodes
        print(f"Average degree: {average_degree}")
        self.distribute_initial_infection(self.number_of_initial_infected)
        day = 1
        plt.figure(figsize=(5, 5))
        while True:
            infected = self.get_by_status("infected")
            self.interact(day)
            self.count_cumulative_cases()
            self.count_daily_cases()
            self.update_disease_progress()
            day += 1
            if not infected:
                break
        plt.close()
        self.plot_cases_curves()

    def interact(self, day, save_steps=False):
        infected = self.get_by_status("infected")
        infected_person_contacts = self.G.edges(infected)
        susceptibles = (person2 for person1,
                        person2 in infected_person_contacts
                        if self.G.nodes[person2]["status"] == "susceptible")

        for susceptible_person in set(susceptibles):
            if self.infection_occurred():
                nx.set_node_attributes(
                    self.G, values={susceptible_person: self.infected})

        if save_steps:
            self.draw(day)

    def update_disease_progress(self):
        def parse_attr(attributes):
            if attributes["days_with_disease"] >= self.d:
                return self.recovered

            attributes["days_with_disease"] += 1
            return attributes

        updated_infected = {person: parse_attr(attributes) for person,
                            attributes in self.G.nodes(data=True)
                            if attributes["status"] == "infected"}
        nx.set_node_attributes(self.G, values=updated_infected)

    def generate_pandemic_network(self, N: int, k: int, p: float) -> nx.Graph:
        self.G = nx.watts_strogatz_graph(N, k, p)
        nx.set_node_attributes(self.G, values="susceptible", name="status")

    def infection_occurred(self):
        return choices([True, False], weights=[self.r, 1 - self.r])[0]

    def distribute_initial_infection(self, number_of_infections: int) -> None:
        whole_network = tuple(self.G.nodes())
        random_infected_people = sample(whole_network, number_of_infections)
        for person in random_infected_people:
            nx.set_node_attributes(self.G, values={person: self.infected})

    def draw(self, day):
        plt.title(f"Day: {day}")
        nodesize = 40

        infected = self.get_by_status("infected")
        susceptibles = self.get_by_status("susceptible")
        recovered = self.get_by_status("recovered")

        pos = nx.circular_layout(self.G)
        nx.draw_networkx_nodes(
            self.G, pos=pos, nodelist=susceptibles, node_color="blue", label="Susceptibles", node_size=nodesize)  # , node_size=50,
        nx.draw_networkx_nodes(
            self.G, pos=pos, nodelist=infected, node_color="red", label="Infected", node_size=nodesize)
        nx.draw_networkx_nodes(
            self.G, pos=pos, nodelist=recovered, node_color="green", label="Recovered", node_size=nodesize)
        nx.draw_networkx_edges(self.G, pos=pos, edge_color="gray")
        plt.legend(scatterpoints=1)
        plt.savefig(f"{day}-evolution.png")
        plt.clf()

    def get_by_status(self, status):
        return [person for person,
                person_status in nx.get_node_attributes(
                    self.G, "status").items()
                if person_status == status]

    def count_cumulative_cases(self):
        self.cumulative_cases.append(len(self.get_by_status(
            "infected")) + len(self.get_by_status("recovered")))

    def count_daily_cases(self):
        today_cases = len([attributes for person, attributes in self.G.nodes(data=True)
                           if attributes["status"] == "infected" and
                           attributes["days_with_disease"] == 1
                           ])
        self.daily_cases.append(today_cases)

    def plot_cases_curves(self):
        _, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

        ax1.set_ylabel("Cumulative cases")
        ax2.set_ylabel("Daily cases")
        ax2.set_xlabel("Days")
        ax1.set_xlim(0, len(self.daily_cases))
        ax1.plot(self.cumulative_cases, color="black")
        ax2.plot(self.daily_cases, color="black")
        plt.savefig(
            f"cases_vs_days_N={self.N}_k={self.k}_p={self.p}_d={self.d}_r={self.r}.png")


Pandemic_Network()
