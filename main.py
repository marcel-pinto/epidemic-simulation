import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import choices, sample


class Pandemic_Network:
    N = 50
    k = 4
    p = 0.5
    d = 6
    number_of_initial_infected = 5
    infected = {
        "status": "infected",
        "days_with_disease": 1
    }
    recovered = {
        "status": "recovered",
    }
    G: nx.Graph
    fig = None

    def __init__(self):
        self.generate_pandemic_network(self.N, self.k, self.p)
        self.distribute_initial_infection(self.number_of_initial_infected)
        day = 1
        self.fig = plt.figure(figsize=(5, 5))
        while True:
            infected = self.get_by_status("infected")
            self.interact(day, save_steps=True)
            self.update_disease_progress()
            day += 1
            if not infected:
                break

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
        return choices([True, False], weights=[0.1, 0.9])[0]

    def distribute_initial_infection(self, number_of_infections: int) -> None:
        whole_network = tuple(self.G.nodes())
        random_infected_people = sample(whole_network, number_of_infections)
        for person in random_infected_people:
            nx.set_node_attributes(self.G, values={person: self.infected})

    def draw(self, day):
        plt.title(f"Day: {day}")
        nodesize = 100

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
        nx.draw_networkx_edges(self.G, pos=pos)
        plt.legend(scatterpoints=1)
        plt.savefig(f"{day}-evolution.png")
        plt.clf()

    def get_by_status(self, status):
        return [person for person,
                person_status in nx.get_node_attributes(
                    self.G, "status").items()
                if person_status == status]


Pandemic_Network()
