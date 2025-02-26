import operator
import datetime
import random
from pydantic import BaseModel, Field
from typing import Annotated, List
from typing_extensions import TypedDict

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


class TripDetails(BaseModel):
    distance_km: float = Field(
        description = "Calculated trip distance in km"
    )
    duration_mins: float = Field(
        description = "Estimated trip duration in minutes"
    )
    prefered_route: str = Field(
        description = "Recommended route based on condition"
    )
    traffic_condition: str = Field(
        description = "Current traffic status"
    )


class PaymentReceipt(BaseModel):
    base_fare: float = Field(
        description = "Initial fare before extras"
    )
    distance_fare: float = Field(
        description = "Cost per kilometer"
    )
    total: float = Field(
        description = "Total amount to pay"
    )


def calculate_dynamic_pricing(state: GeneralInfoAboutTrip):
    """Calculate fare based on distance and traffic conditions"""
    base_fare = 3.00 # initial charge
    per_km = 1.50 # cost per km
    night_surcharge = 1.20 # 20% surcharge between 20:00-06:00

    distance = _simulate_distance_calculation(
        state["heading_from"],
        state["destination"]
    )

    current_hour = datetime.datetime.now().hour
    time_multiplier = night_surcharge if 20 <= current_hour <= 24 or 0 <= current_hour < 6 else 1.0
    total = (base_fare + (distance * per_km)) * time_multiplier

    return {
        "approximate_cost": round(total, 2),
        "distance_km": distance,
        "time_multiplier": time_multiplier
    }

def _simulate_distance_calculation(start: str, end: str) -> float:
    """simulate distance calculation between two points"""
    # just random simulation for now
    return random.uniform(2.0, 25.0)

def suggest_routes(state: GeneralInfoAboutTrip):
    """ suggest optimal routes with traffic awareness"""
    routes = [
        {"name": "Fastest", "time": "25 mins", "traffic": "light"},
        {"name": "Scenic", "time": "35 mins", "traffic": "moderate"},
        {"name": "Economy", "time": "30 mins", "traffic": "moderate"}
    ]

    best_route = min(routes, key=lambda x: int(x["time"].split()[0]))

    return {
        "prefered_route": best_route["name"],
        "estimated_duration": best_route["time"],
        "traffic_condition": best_route["traffic"]
    }

def process_payment(state: GeneralInfoAboutTrip):
    """simulate payment processing"""
    payment_methods = ["Credit Card", "Cash", "Mobile Pay"]
    total = state["approximate_cost"]

    return {
        "payment_status": "completed",
        "payment_method": random.choice(payment_methods),
        "receipt": {
            "base_fare": 3.00,
            "distance_fare": round(total - 3.00, 2),
            "total": total
        }
    }

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
    """get human validation for critical decisions"""
    print(f"suggested route: {state['prefered_route']}")
    print(f"estimated cost: ${state['approximate_cost']}")
    response = input("Approve trip details? (Y/n): ")

    return {
        "trip_approved": response.lower() == "y",
        "human_approve": response
    }


# Add nodes and edges
builder = StateGraph(ResearchAboutTrip)
builder.add_node("create_taxi_driver", create_driver)
builder.add_node("calculate_pricing", calculate_dynamic_pricing)
builder.add_node("suggest_routes", suggest_routes)
builder.add_node("process_payment", process_payment)
builder.add_node("human_feedback", human_feedback)

# Logic
builder.set_entry_point("create_taxi_driver")
builder.add_edge("create_taxi_driver", "calculate_pricing")
builder.add_edge("calculate_pricing", "suggest_routes")
builder.add_edge("suggest_routes", "process_payment")
builder.add_edge("process_payment", "human_feedback")
builder.add_edge("human_feedback", END)


class EnchancedTaxiDriver(TaxiDriver):
    conversation_history: List[str] = Field(
        default_factory = list,
        description = "Remember key points from current conversation"
    )
    passenger_preferences: dict = Field(
        default_factory = dict,
        description = "remember passenger preferences and choices"
    )

def update_conversation_memory(state: MessagesState):
    """Maintain conversation context"""
    last_message = state["messages"][-1].content
    driver = state["taxi_driver"][0]

    if "prefers" in last_message or "like to" in last_message:
        driver.passenger_preferences.update(
            _extract_preferences(last_message)
        )

    driver.conversation_history.append(last_message)
    if len(driver.conversation_history) > 3:
        driver.conversation_history.pop(0)

    return {"taxi_driver": [driver]}

def _extract_preferences(text: str) -> dict:
    """extract passenger preferences from conversation"""
    return {
        "music_preference": "classic" if "classic" in text else "modern",
        "conversation_style": "quiet" if "quiet" in text else "talkative"
    }

graph = builder.compile(interrupt_before=["human_feedback"])
