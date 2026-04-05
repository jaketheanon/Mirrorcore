# Memory Model Design

> **Note:** This document describes design intent and conceptual layers. Canonical table definitions and migrations live under `src/mirrorcore/db/`; use the code and tests as the source of truth for what is persisted today.

## Overview

The memory system provides persistent storage and retrieval of episodic interactions, patterns, and learned insights. It enables Mirrorcore to provide contextually relevant assistance based on historical user interactions.

## Memory Architecture

### 1. Memory Types

#### Episodic Memory
- **Session Logs**: Complete interaction transcripts with metadata
- **Problem-Solving Episodes**: Troubleshooting sessions with outcomes
- **Decision Records**: Decision contexts and chosen solutions
- **Learning Events**: Pattern discovery and model updates

#### Semantic Memory
- **Conceptual Knowledge**: Generalized patterns and rules
- **Procedural Knowledge**: Step-by-step solution approaches
- **Factual Knowledge**: Technical facts and configurations
- **Pattern Libraries**: Reusable solution templates

#### Working Memory
- **Session Context**: Current interaction state and history
- **Active Patterns**: Recently relevant problem-solving approaches
- **Temporary State**: Short-term context for ongoing tasks
- **Buffer Data**: Pending information for processing

### 2. Data Structures

#### Memory Entry Schema
```json
{
  "id": "uuid",
  "type": "episode|pattern|knowledge",
  "timestamp": "ISO datetime",
  "context": {
    "domain": "terminal|decision|learning",
    "environment": "development|production|testing",
    "complexity": "simple|moderate|complex"
  },
  "content": {
    "input": "user input or problem description",
    "process": "reasoning steps taken",
    "output": "solution or response provided",
    "outcome": "success|failure|partial"
  },
  "metadata": {
    "confidence": 0.0-1.0,
    "relevance": 0.0-1.0,
    "access_count": integer,
    "last_accessed": "ISO datetime"
  },
  "relationships": [
    {
      "type": "similar_to|caused_by|led_to",
      "target_id": "uuid",
      "strength": 0.0-1.0
    }
  ]
}
```

#### Pattern Schema
```json
{
  "id": "uuid",
  "name": "descriptive pattern name",
  "category": "troubleshooting|decision|communication",
  "description": "pattern description",
  "conditions": [
    {
      "field": "error_type|context|environment",
      "operator": "equals|contains|matches",
      "value": "condition value"
    }
  ],
  "actions": [
    {
      "step": integer,
      "action": "command|explanation|question",
      "content": "action content",
      "success_rate": 0.0-1.0
    }
  ],
  "metadata": {
    "usage_count": integer,
    "success_rate": 0.0-1.0,
    "last_updated": "ISO datetime",
    "confidence": 0.0-1.0
  }
}
```

### 3. Memory Operations

#### Storage Operations (`memory/updater.py`)
```python
class MemoryUpdater:
    def store_episode(self, episode: Episode) -> str
    def update_pattern(self, pattern: Pattern) -> bool
    def create_relationship(self, source_id: str, target_id: str, type: str) -> bool
    def increment_access(self, memory_id: str) -> bool
    def update_confidence(self, memory_id: str, confidence: float) -> bool
```

#### Retrieval Operations (`memory/retrieval.py`)
```python
class MemoryRetrieval:
    def search_episodes(self, query: Query) -> List[Episode]
    def find_patterns(self, context: Context) -> List[Pattern]
    def get_similar_situations(self, current: Situation) -> List[Episode]
    def retrieve_by_timeframe(self, start: datetime, end: datetime) -> List[Episode]
    def get_successful_solutions(self, problem_type: str) -> List[Pattern]
```

#### Extraction Operations (`memory/extractor.py`)
```python
class MemoryExtractor:
    def extract_patterns(self, episodes: List[Episode]) -> List[Pattern]
    def identify_insights(self, interactions: List[Interaction]) -> List[Insight]
    def calculate_success_rates(self, episodes: List[Episode]) -> Dict[str, float]
    def discover_relationships(self, memories: List[Memory]) -> List[Relationship]
```

### 4. Retrieval Strategies

#### Similarity-Based Retrieval
- **Text Similarity**: Vector similarity of problem descriptions
- **Context Similarity**: Environment and domain matching
- **Pattern Similarity**: Structural and procedural similarity
- **Outcome Similarity**: Success/failure pattern matching

#### Temporal Retrieval
- **Recency Weighting**: More recent memories weighted higher
- **Temporal Patterns**: Time-based problem occurrence patterns
- **Seasonal Variations**: Context changes over time periods
- **Evolution Tracking**: User learning and adaptation over time

#### Relevance Scoring
```python
def calculate_relevance(memory: Memory, query: Query) -> float:
    text_similarity = cosine_similarity(memory.content, query.content)
    context_match = calculate_context_similarity(memory.context, query.context)
    recency_bonus = calculate_recency_bonus(memory.timestamp)
    success_weight = memory.metadata.confidence * memory.outcome.success_rate
    
    return weighted_average([text_similarity, context_match, recency_bonus, success_weight])
```

### 5. Pattern Learning

#### Pattern Discovery
- **Sequence Mining**: Identify common action sequences
- **Association Rules**: Discover correlated conditions and solutions
- **Clustering**: Group similar problem-solving approaches
- **Anomaly Detection**: Identify unusual but effective solutions

#### Pattern Validation
- **Cross-Validation**: Test patterns on held-out data
- **Success Rate Tracking**: Monitor pattern effectiveness
- **User Feedback**: Incorporate explicit pattern validation
- **Confidence Scoring**: Assign reliability scores to patterns

#### Pattern Evolution
- **Incremental Learning**: Update patterns with new data
- **Pattern Merging**: Combine similar patterns
- **Pattern Pruning**: Remove ineffective or outdated patterns
- **Pattern Specialization**: Create specific variants of general patterns

### 6. Memory Management

#### Storage Optimization
- **Compression**: Compress old or rarely accessed memories
- **Indexing**: Efficient database indexes for common queries
- **Caching**: Cache frequently accessed patterns and episodes
- **Archival**: Move old memories to long-term storage

#### Privacy Controls
- **Data Minimization**: Store only necessary information
- **Retention Policies**: Automatic cleanup of old data
- **Access Controls**: Restrict sensitive memory access
- **Encryption**: Protect sensitive memory contents

#### Data Lifecycle
```python
class MemoryLifecycle:
    def create_memory(self, content: dict) -> Memory
    def activate_memory(self, memory_id: str) -> bool
    def archive_memory(self, memory_id: str) -> bool
    def delete_memory(self, memory_id: str) -> bool
    def cleanup_expired(self) -> int
```

### 7. Performance Optimization

#### Query Optimization
- **Index Strategy**: Optimize database indexes for common queries
- **Query Caching**: Cache results of expensive queries
- **Lazy Loading**: Load memory details only when needed
- **Batch Operations**: Process multiple memories efficiently

#### Memory Efficiency
- **Pattern Compression**: Represent patterns compactly
- **Deduplication**: Remove duplicate or similar memories
- **Selective Retention**: Keep only high-value memories
- **Memory Monitoring**: Track memory usage and performance

## Integration Points

### Agent Integration
- **Intake Agent**: Store assessment results and initial patterns
- **Terminal Agent**: Record troubleshooting sessions and solutions
- **Decision Agent**: Log decision contexts and outcomes
- **Persona Agent**: Update user model based on interactions

### LLM Integration
- **Context Enhancement**: Use memories to enrich LLM prompts
- **Response Validation**: Compare LLM suggestions with historical patterns
- **Learning Integration**: Extract patterns from LLM interactions
- **Fallback Support**: Use memories when LLM unavailable

## Evaluation Metrics

### Memory Quality
- **Retrieval Accuracy**: Relevance of retrieved memories
- **Pattern Effectiveness**: Success rate of pattern-based suggestions
- **Coverage**: Breadth of problem types covered
- **Freshness**: Currency of stored information

### System Performance
- **Retrieval Speed**: Time to find relevant memories
- **Storage Efficiency**: Memory usage per stored item
- **Update Performance**: Speed of memory updates
- **Query Throughput**: Queries per second capacity

## Future Enhancements

### Advanced Learning
- **Transfer Learning**: Apply patterns across domains
- **Meta-Learning**: Learn how to learn better patterns
- **Hierarchical Patterns**: Multi-level pattern abstraction
- **Causal Reasoning**: Understand cause-effect relationships

### External Integration
- **Documentation Integration**: Import knowledge from external docs
- **Community Patterns**: Share anonymized patterns with users
- **Tool Integration**: Learn from tool usage patterns
- **Workflow Integration**: Integrate with development workflows
