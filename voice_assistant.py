import operator
from pydantic import BaseModel, Field
from typing import Annotated, List
from typing_extensions import TypedDict

from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string
from langchain_openai import ChatOpenAI

from langgraph.constants import Send
from langgraph.graph import END, MessagesState, START, StateGraph


llm = ChatOpenAI(model="gpt-4o", temperature=0)


class TaxiDriver(BaseModel):
    name: str = Field(
            description="Name of the driver"
    )
    description: str = Field(
            description="Description of the driver focus, concerns and motives"
    )

    @property
    def person(self) -> str:
        return f"Name: {self.name}\nDescription: {self.description}\n"


class Perspectives(BaseModel):
    approximate_cost: int = Field(
            description = "approximate cost of the trip"
    )
    taxi_driver: List[TaxiDriver] = Field(
            description = "list of driver or drivers with their roles"
    )


class GeneralInfoAboutTrip(TypedDict):
    destination: str       # where are we driving to
    heading_from: str      # location where pick client up
    human_approve: str     # approving, what our trip points are correct
    approximate_cost: int  # how much it would cost
    taxi_driver: List[TaxiDriver]


class SearchQueryForTrip(BaseModel):
    search_query: str = Field(None, description="search query for information retrieval")


class ResearchAboutTrip(TypedDict):
    destination: str
    heading_from: str
    human_approve: str
    approximate_cost: int
    total_cost: int


taxi_driver_instructions = """You are an AI agent roleplaying as a taxi driver. Follow these instructions:

1. Understand the role:
   - You are a taxi driver with a unique personality.
   - Your personality may include humor, sarcasm, friendliness, or quiet professionalism.
   - Engage passengers in conversation but adapt to their preferences.
   - base on destination - {destination} and heading from {heading_from} randomly generate cost of the trip
   - based on editorial feedback that has been optionally provided to guide creation of the taxi driver: {human_approve}

2. Analyze the conversation context:
   - Pay attention to the passenger's mood and tone.
   - Keep responses concise but expressive, adjusting your style to match the situation.
   - Use short-term memory to recall details from the current session.

3. Respond according to the role:
   - If the passenger prefers silence, respect their choice.
   - If they are talkative, engage them with interesting topics or witty remarks.
   - If they are nervous, use humor or reassurance to make them comfortable.

4. Maintain authenticity:
   - Occasionally share taxi driver stories (generated dynamically).
   - Use regional slang or phrases to make conversations more natural. Adapt to regional info based on destination -
   {destination} and {heading_from} locations, but its not strictly nessesary, if u cant recognize - then follow it.
   - Avoid excessive repetition or breaking character.

5. Manage memory and voice response:
   - Remember key details from the current conversation.
   - Respond in a tone that fits the assigned personality—gruff, cheerful, mysterious, etc.

6. Prepare for future enhancements:
   - In future versions, you may provide navigation, suggest routes, and track trips.
"""

def create_driver(state: GeneralInfoAboutTrip):
    """ Creates random taxi driver """
    destination = state['destination']
    heading_from = state['heading_from']
    # approximate_cost = state['approximate_cost']
    human_approve = state.get('human_approve', '')

    structured_llm = llm.with_structured_output(Perspectives)
    system_message = taxi_driver_instructions.format(
            destination=destination,
            heading_from=heading_from,
            # approximate_cost=approximate_cost,
            human_approve=human_approve
    )
    taxi_driver = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content="Generate taxi driver.")])

    return {"taxi_driver": taxi_driver.taxi_driver, "approximate_cost": taxi_driver.approximate_cost}

def human_feedback(state: GeneralInfoAboutTrip):
    """ node that should be interrupted on """
    pass


# Add nodes and edges
builder = StateGraph(ResearchAboutTrip)
builder.add_node("create_taxi_driver", create_driver)
builder.add_node("human_feedback", human_feedback)

# Logic
builder.add_edge(START, "create_taxi_driver")
builder.add_edge("create_taxi_driver", "human_feedback")
builder.add_edge("human_feedback", END)

graph = builder.compile(interrupt_before=["human_feedback"])
