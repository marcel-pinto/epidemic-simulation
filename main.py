import networkx as nx
import matplotlib.pyplot as plt
from random import choices, sample
from poisson_small_world_network import poisson_small_world_graph
import numpy as np
from typing import TypedDict, Optional

Parameters = TypedDict('Parameters',
                       {
                           "number_of_nodes": int,
                           "D": int,
                           "epsilon": float,
                           "infection_rate": float,
                           "days_infectious": int,
                           "number_of_initial_infected": int,
                       })


NpiParameters = TypedDict('NpiParameters',
                          {
                              "D": int,
                              "epsilon": float,
                              "infection_rate": float,
                              "days_infectious": float
                          },
                          total=False
                          )


class Epidemic_Network:
    infected = {
        "status": "infected",
        "days_with_disease": 1
    }
    recovered = {
        "status": "recovered",
    }
    G: nx.Graph

    def __init__(self,
                 parameters: Parameters,
                 npi_parameters: NpiParameters = None,
                 max_days=100,
                 save_steps=False,
                 ):

        self.parameters = parameters
        self.npi_parameters = npi_parameters

        self.daily_cases = []

        self.generate_epidemic_network(
            self.parameters["number_of_nodes"],
            self.parameters["D"],
            self.parameters["epsilon"])
        self.distribute_initial_infection(
            self.parameters["number_of_initial_infected"])
        if save_steps:
            plt.figure(figsize=(5, 5))

        day = 1
        while day <= max_days:
            self.count_daily_cases()
            self.update_disease_progress()

            # infected = self.get_by_status("infected")
            self.interact(day, save_steps)
            day += 1
            if self.npi_parameters and self.npi_parameters['npi start at day'] == day:
                print(f"Applying NPI at day {day}")
                self.__apply_NPI()
            # if not infected:
            #     break
        plt.close()

    def generate_epidemic_network(self, number_of_nodes, D, epsilon) -> nx.Graph:
        self.G = poisson_small_world_graph(number_of_nodes, D, epsilon)
        nx.set_node_attributes(self.G, values="susceptible", name="status")

    def interact(self, day, save_steps=False):
        if save_steps:
            self.draw(day)
        infected = self.get_by_status("infected")
        infected_person_contacts = self.G.edges(infected)
        susceptibles = (person2 for person1,
                        person2 in infected_person_contacts
                        if self.G.nodes[person2]["status"] == "susceptible")

        for susceptible_person in susceptibles:
            if self.infection_occurred():
                nx.set_node_attributes(
                    self.G, values={susceptible_person: self.infected})

    def update_disease_progress(self):
        def parse_attr(attributes):
            if attributes["days_with_disease"] >= self.parameters["days_infectious"]:
                attributes.pop("days_with_disease", None)
                return self.recovered

            attributes["days_with_disease"] += 1
            return attributes

        updated_infected = {person: parse_attr(attributes) for person,
                            attributes in self.G.nodes(data=True)
                            if attributes["status"] == "infected"}
        nx.set_node_attributes(self.G, values=updated_infected)

    def infection_occurred(self):
        return choices([True, False],
                       weights=[self.parameters["infection_rate"],
                                1 - self.parameters["infection_rate"]])[0]

    def distribute_initial_infection(self, number_of_infections: int) -> None:
        whole_network = tuple(self.G.nodes())
        random_infected_people = sample(whole_network, number_of_infections)
        for person in random_infected_people:
            nx.set_node_attributes(self.G, values={person: self.infected})

    def get_by_status(self, status):
        return [person for person,
                person_status in nx.get_node_attributes(
                    self.G, "status").items()
                if person_status == status]

    def count_daily_cases(self):
        infected_people = nx.get_node_attributes(self.G, "days_with_disease")
        number_of_days_with_infection = list(infected_people.values())
        self.daily_cases.append(number_of_days_with_infection.count(1))

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

    def get_daily_cases(self):
        return self.daily_cases

    def __make_structural_changes(self):
        self.G = poisson_small_world_graph(
            self.G.number_of_nodes(),
            self.parameters['D'],
            self.parameters['epsilon'],
            self.G)

    def __apply_NPI(self):
        self.parameters = {**self.parameters, **self.npi_parameters}
        if any(k in self.npi_parameters for k in ('D', 'epsilon')):
            self.__make_structural_changes()


if __name__ == "__main__":

    def plot(n, D, epsilon, r, d, i0):
        network = Epidemic_Network(
            parameters={
                "number_of_nodes": n,
                "D": D,
                "epsilon": epsilon,
                "infection_rate": r,
                "days_infectious": d,
                "number_of_initial_infected": i0,
            },
            npi_parameters={
                "npi start at day": 10,
                "D": 8,
                "epsilon": 0
            }
        )

        daily_cases = network.get_daily_cases()
        cumulative_cases = np.cumsum(daily_cases)

        _, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, dpi=300)

        ax1.set_ylabel("Cumulative cases")
        ax2.set_ylabel("Daily cases")
        ax2.set_xlabel("Days")
        ax1.plot(cumulative_cases, color="black", label="Network")
        ax2.bar(list(range(len(daily_cases))),
                daily_cases, color="red", edgecolor="black")
        ax1.set_title(
            f"N={n}, D={D}, r={r}, $\\epsilon={epsilon}$")

        ax1.legend()
        sulfix = f"type=poisson_small_world_graph_n={n}_D={D}_epsilon={epsilon}_r={r}_d={d}_i0={i0}"
        filename = f"cases_vs_days_{sulfix}.png"
        plt.savefig(filename)
        plt.close()

    plot(n=1000, D=3, epsilon=0.3, r=0.1, d=6, i0=10)
