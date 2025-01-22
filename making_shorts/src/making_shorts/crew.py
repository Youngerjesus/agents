from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

from langchain_community.llms import OpenAI
from crewai import LLM 

# OpenAI 4o
# llm = OpenAI(model="gpt-4o", temperature=1, max_tokens=4096)

# Anthropic Sonnet 3.5
llm = LLM(model="anthropic/claude-3-5-sonnet-20241022", temperature=1, max_tokens=4096, top_p=0.9)


@CrewBase
class MakingShorts():
	"""MakingShorts crew"""

	# Learn more about YAML configuration files here:
	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools
	@agent
	def scenario_creator(self) -> Agent:
		return Agent(
			config=self.agents_config['scenario_creator'],
			verbose=True,
			llm=llm,
		)
  
	@task
	def dramatic_narration_task(self) -> Task:
		return Task(
			config=self.tasks_config['dramatic_narration_task'],
			verbose=True,
			output_file='output/dramatic_narration_task.txt',
		)
  
	@agent
	def story_analysis_and_growth_enhancement_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['story_analysis_and_growth_enhancement_agent'],
			verbose=True,
			llm=llm,
		)
  
	@agent
	def story_polishing_and_improvement_navigator(self) -> Agent: 
		return Agent(
			config=self.agents_config['story_polishing_and_improvement_navigator'],
			verbose=True,
			llm=llm,
		)
  
	@task
	def feedback_task(self) -> Task:
		return Task(
			config=self.tasks_config['feedback_task'],
			verbose=True,
		)
	
	@task
	def refinement_task(self) -> Task:
		return Task(
			config=self.tasks_config['refinement_task'],
			verbose=True,
		)
 
	@crew
	def crew(self) -> Crew:
		"""Creates the MakingShorts crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
