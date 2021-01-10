import networkx as nx
import matplotlib.pyplot as plt
from random import choices, sample
from poisson_small_world_network import poisson_small_world_graph
from sir import sir_model_dynamics
import numpy as np


class Epidemic_Network:
    infected = {
        "status": "infected",
        "days_with_disease": 1
    }
    recovered = {
        "status": "recovered",
    }
    G: nx.Graph
    cumulative_cases = []
    daily_cases = []

    D = None
    k = None
    p = None

    def __init__(self,
                 infection_rate,
                 days_until_recovery,
                 number_of_initial_infected,
                 save_steps=False
                 ):
        self.infection_rate = infection_rate
        self.days_until_recovery = days_until_recovery
        self.number_of_initial_infected = number_of_initial_infected
        self.daily_cases.append(number_of_initial_infected)
        self.save_steps = save_steps

    def generate_epidemic_network(self, network_generation_method, **kwargs) -> nx.Graph:
        self.G = network_generation_method(**kwargs)

        if "D" in kwargs.keys():
            self.D = kwargs["D"]
        if "k" in kwargs.keys():
            self.k = kwargs["k"]
        self.p = kwargs["p"]

        self.network_gen_method = network_generation_method.__name__
        self.network_gen_parameters = str(kwargs)
        nx.set_node_attributes(self.G, values="susceptible", name="status")
        return self

    def simulate(self):
        self.distribute_initial_infection(self.number_of_initial_infected)
        day = 1
        if self.save_steps:
            plt.figure(figsize=(5, 5))
        while True:
            # infected = self.get_by_status("infected")
            self.interact(day, self.save_steps)
            self.count_cumulative_cases()
            self.count_daily_cases()
            self.update_disease_progress()
            day += 1
            if day > 100:
                break
                # if not infected:
                #     break
        plt.close()
        self.plot_cases()

    def interact(self, day, save_steps=False):
        infected = self.get_by_status("infected")
        infected_person_contacts = self.G.edges(infected)
        susceptibles = (person2 for person1,
                        person2 in infected_person_contacts
                        if self.G.nodes[person2]["status"] == "susceptible")

        for susceptible_person in susceptibles:
            if self.infection_occurred():
                nx.set_node_attributes(
                    self.G, values={susceptible_person: self.infected})

        if save_steps:
            self.draw(day)

    def update_disease_progress(self):
        def parse_attr(attributes):
            if attributes["days_with_disease"] >= self.days_until_recovery:
                return self.recovered

            attributes["days_with_disease"] += 1
            return attributes

        updated_infected = {person: parse_attr(attributes) for person,
                            attributes in self.G.nodes(data=True)
                            if attributes["status"] == "infected"}
        nx.set_node_attributes(self.G, values=updated_infected)

    def infection_occurred(self):
        return choices([True, False],
                       weights=[self.infection_rate, 1 - self.infection_rate])[0]

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

    def count_cumulative_cases(self):
        self.cumulative_cases.append(len(self.get_by_status(
            "infected")) + len(self.get_by_status("recovered")))

    def count_daily_cases(self):
        today_cases = len([attributes for person, attributes in self.G.nodes(data=True)
                           if attributes["status"] == "infected" and
                           attributes["days_with_disease"] == 1
                           ])
        self.daily_cases.append(today_cases)

    def plot_cases(self):
        _, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, dpi=300)

        ax1.set_ylabel("Cumulative cases")
        ax2.set_ylabel("Daily cases")
        ax2.set_xlabel("Days")
        # ax1.set_xlim(-0.1, len(self.daily_cases))
        ax1.plot(self.cumulative_cases, color="black", label="Network")
        # ax2.plot(self.daily_cases, 'black', marker="o")
        ax2.bar(list(range(len(self.daily_cases))),
                self.daily_cases, color="red", edgecolor="black")
        if self.D:
            ax1.set_title(
                f"N={self.G.number_of_nodes()}, D={self.D}, r={self.infection_rate}, $\\epsilon={self.p}$")
            beta = self.infection_rate * self.D
        if self.k:
            ax1.set_title(
                f"N={self.G.number_of_nodes()}, k={self.k}, r={self.infection_rate}, $\\epsilon={self.p}$")
            beta = self.infection_rate * self.k

        gamma = 1/self.days_until_recovery
        t = np.linspace(0, 100, 100)
        _, I, R = sir_model_dynamics(
            N=self.G.number_of_nodes(),
            I0=self.number_of_initial_infected,
            beta=beta,
            gamma=gamma,
            t=t
        )
        ax1.plot(R + I, 'g', label="Recovered + Infected (SIR)")
        ax1.legend()
        sulfix = self.parse_output_file_name_sulfix()
        filename = f"cases_vs_days_{sulfix}.png"
        plt.savefig(filename)
        plt.clf()

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

    def parse_output_file_name_sulfix(self):
        self.network_gen_parameters = self.network_gen_parameters.replace(
            "{", "").replace("}", "").replace(", ", "_") \
            .replace(": ", "=").replace("'", "")
        return f"type={self.network_gen_method}_{self.network_gen_parameters}_r={self.infection_rate}_d={self.days_until_recovery}_i0={self.number_of_initial_infected}"


# Epidemic_Network(
#     infection_rate=0.1,
#     days_until_recovery=6,
#     number_of_initial_infected=10,
# ).generate_epidemic_network(
#     nx.watts_strogatz_graph, n=1000, k=10, p=0.1).simulate()


Epidemic_Network(
    infection_rate=0.1,
    days_until_recovery=6,
    number_of_initial_infected=10,
).generate_epidemic_network(
    poisson_small_world_graph, n=1000, D=3, p=0.1).simulate()
