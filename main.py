import networkx as nx
import matplotlib.pyplot as plt
from random import choices, sample
from .poisson_small_world_network import poisson_small_world_graph
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

    def __init__(self,
                 infection_rate,
                 days_until_recovery,
                 number_of_initial_infected,
                 number_of_nodes,
                 D,
                 epsilon,
                 max_days=100,
                 save_steps=False,
                 ):
        self.infection_rate = infection_rate
        self.days_until_recovery = days_until_recovery

        self.daily_cases = []

        self.generate_epidemic_network(number_of_nodes, D, epsilon)
        self.distribute_initial_infection(number_of_initial_infected)
        if save_steps:
            plt.figure(figsize=(5, 5))

        day = 1
        while day <= max_days:
            self.count_daily_cases()
            self.update_disease_progress()

            # infected = self.get_by_status("infected")
            self.interact(day, save_steps)
            day += 1
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
            if attributes["days_with_disease"] >= self.days_until_recovery:
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


if __name__ == "__main__":
    from sir import sir_model_dynamics

    def plot(n, D, epsilon, r, d, i0):
        network = Epidemic_Network(
            number_of_nodes=n,
            D=D,
            epsilon=epsilon,
            infection_rate=r,
            days_until_recovery=d,
            number_of_initial_infected=i0
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

        beta = r * D
        gamma = 1/d

        t = np.linspace(0, 100, 100)
        _, I, R = sir_model_dynamics(
            N=n,
            I0=i0,
            beta=beta,
            gamma=gamma,
            t=t
        )
        ax1.plot(R, 'g', label="Recovered + Infected (SIR)")
        ax1.legend()
        sulfix = f"type=poisson_small_world_graph_n={n}_D={D}_epsilon={epsilon}_r={r}_d={d}_i0={i0}"
        filename = f"cases_vs_days_{sulfix}.png"
        plt.savefig(filename)
        plt.close()

        with open(f'daily_cases_{sulfix}.txt', 'w') as f:
            for day, cases in enumerate(daily_cases):
                f.write(f"{day} {cases}\n")

        with open(f'cumulative_cases_{sulfix}.txt', 'w') as f:
            for day, cases in enumerate(cumulative_cases):
                f.write(f"{day} {cases}\n")
    plot(n=1000, D=3, epsilon=0.1, r=0.1, d=6, i0=10)
    plot(n=1000, D=8, epsilon=0.1, r=0.1, d=6, i0=10)
