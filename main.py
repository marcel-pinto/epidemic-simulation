# pylint: disable=relative-beyond-top-level,
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from typing import TypedDict, Optional
from network_generator import watts_strogatz_clique_graph


r = TypedDict(
    "r",
    {
        "intra": float,
        "inter": float,
    },
)

Parameters = TypedDict(
    "Parameters",
    {
        "n": int,
        "D": int,
        "epsilon": float,
        "r": r,
        "d": int,
        "i0": int,
        "household_distribution": list,
    },
)


NpiParameters = TypedDict(
    "NpiParameters",
    {
        "D": int,
        "epsilon": float,
        "r": r,
        "d": float,
        "household_distribution": list,
        "npi_start_day": int,
    },
    total=False,
)


class Epidemic_Network:
    infected = {"status": "infected", "days_with_disease": 1}
    recovered = {
        "status": "recovered",
    }
    G: nx.Graph

    def __init__(
        self,
        parameters: Parameters,
        npi_parameters: NpiParameters = None,
        max_days=100,
        save_steps=False,
        random_state=None,
    ):

        self.parameters = parameters
        self.npi_parameters = npi_parameters

        self.random_state = random_state
        self.daily_cases = []

        random.setstate(random_state)

        if "household_distribution" not in self.parameters:
            self.generate_epidemic_network(
                self.parameters["n"], self.parameters["D"], self.parameters["epsilon"]
            )
        else:
            self.generate_epidemic_network(
                self.parameters["household_distribution"],
                self.parameters["D"],
                self.parameters["epsilon"],
            )

        self.distribute_initial_infection(self.parameters["i0"])
        if save_steps:
            plt.figure(figsize=(5, 5))

        day = 1
        while day <= max_days:
            self.count_daily_cases()
            self.update_disease_progress()
            self.interact(day, save_steps)
            day += 1
            if self.npi_parameters and self.npi_parameters["npi_start_day"] == day:
                self.__apply_NPI()
        plt.close()

    def generate_epidemic_network(self, n: int or list, k: int, p: float) -> nx.Graph:
        self.G = watts_strogatz_clique_graph(n, k, p)
        nx.set_node_attributes(self.G, values="susceptible", name="status")

    def interact(self, day, save_steps=False):
        if save_steps:
            self.draw(day)
        infected = self.get_by_status("infected")
        infected_person_contacts = self.G.edges(infected, data=True)
        susceptibles = (
            (person2, connection["connection_type"])
            for person1, person2, connection in infected_person_contacts
            if self.G.nodes[person2]["status"] == "susceptible"
        )

        for susceptible_person, connection_type in susceptibles:
            if self.infection_occurred(connection_type):
                nx.set_node_attributes(self.G, values={susceptible_person: self.infected})

    def update_disease_progress(self):
        def parse_attr(attributes):
            if attributes["days_with_disease"] >= self.parameters["d"]:
                attributes.pop("days_with_disease", None)
                return self.recovered

            attributes["days_with_disease"] += 1
            return attributes

        updated_infected = {
            person: parse_attr(attributes)
            for person, attributes in self.G.nodes(data=True)
            if attributes["status"] == "infected"
        }
        nx.set_node_attributes(self.G, values=updated_infected)

    def infection_occurred(self, connection_type: str) -> bool:
        return random.choices(
            [True, False],
            weights=[
                self.parameters["r"][connection_type],
                1 - self.parameters["r"][connection_type],
            ],
        )[0]

    def distribute_initial_infection(self, number_of_infections: int) -> None:
        whole_network = tuple(self.G.nodes())
        random_infected_people = random.sample(whole_network, number_of_infections)
        for person in random_infected_people:
            nx.set_node_attributes(self.G, values={person: self.infected})

    def get_by_status(self, status):
        return [
            person
            for person, person_status in nx.get_node_attributes(self.G, "status").items()
            if person_status == status
        ]

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

        pos = nx.spring_layout(self.G, seed=10)
        nx.draw_networkx_nodes(
            self.G,
            pos=pos,
            nodelist=susceptibles,
            node_color="blue",
            label="Susceptibles",
            node_size=nodesize,
        )  # , node_size=50,

        nx.draw_networkx_nodes(
            self.G,
            pos=pos,
            nodelist=infected,
            node_color="red",
            label="Infected",
            node_size=nodesize,
        )

        nx.draw_networkx_nodes(
            self.G,
            pos=pos,
            nodelist=recovered,
            node_color="green",
            label="Recovered",
            node_size=nodesize,
        )

        nx.draw_networkx_edges(self.G, pos=pos, edge_color="gray")
        plt.legend(scatterpoints=1)
        plt.savefig(f"{day}-evolution.png")
        plt.clf()

        return pos

    def get_daily_cases(self):
        return self.daily_cases

    def __make_structural_changes(self):
        H = nx.watts_strogatz_graph(
            self.G.number_of_nodes(), self.parameters["D"], self.parameters["epsilon"]
        )

        new_edges = H.edges()
        old_edges = self.G.edges()
        self.G.remove_edges_from(old_edges)
        self.G.add_edges_from(new_edges)

    def __apply_NPI(self):
        self.parameters = {**self.parameters, **self.npi_parameters}
        if any(k in self.npi_parameters for k in ("D", "epsilon")):
            self.__make_structural_changes()

    def get_random_state(self):
        return random.getstate()


if __name__ == "__main__":
    from utils import generate_distribution_from_hist

    def plot(D, epsilon, r_intra, r_inter, d, i0, hist=np.array([]), n=None):

        parameters = {
            "D": D,
            "epsilon": epsilon,
            "r": {"intra": r_intra, "inter": r_inter},
            "d": d,
            "i0": i0,
        }

        if n:
            parameters["n"] = n
            sulfix = f"n={n}_k={D}_epsilon={epsilon}_r={r_inter}_d={d}_i0={i0}"
        if hist.size > 0:
            dist = generate_distribution_from_hist(hist)
            parameters["household_distribution"] = dist
            sulfix = (
                f"n={sum(dist)}_k={D}_epsilon={epsilon}"
                f"_r_inter={r_inter}_r_intra={r_intra}_d={d}_i0={i0}"
            )

        network = Epidemic_Network(
            parameters=parameters,
            # npi_parameters={"npi_start_day": 20, "D": 2, "epsilon": 0, "r": 0, "d": 6},
        )

        _, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, dpi=300)
        ax1.set_ylabel("Cumulative cases")
        ax2.set_ylabel("Daily cases")
        ax2.set_xlabel("Days")

        daily_cases = network.get_daily_cases()
        cumulative_cases = np.cumsum(daily_cases)

        ax1.plot(cumulative_cases, color="black", label="Network")
        ax2.bar(list(range(len(daily_cases))), daily_cases, color="red", edgecolor="black")

        ax1.legend()

        filename = f"cases_vs_days_{sulfix}.png"
        plt.savefig(filename)
        plt.close()

    hist = np.array(
        [
            [0, 0],
            [1, 2879],
            [2, 1724],
            [3, 1278],
            [4, 849],
            [5, 345],
            [6, 177],
            [7, 66],
            [8, 25],
            [9, 9],
            [10, 6],
            [11, 3],
            [12, 1],
            [13, 1],
        ]
    )

    # hist = np.array([[5, 20]])

    plot(hist=hist, D=6, epsilon=0.3, r_intra=0.5, r_inter=0.1, d=6, i0=100)
