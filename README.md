# kanuni-openai

**OpenAI Chat Completions API Formatter for Kanuni**

Converts [Kanuni](https://www.npmjs.com/package/kanuni) queries into OpenAI Chat Completions API parameters. Supports both OpenAI and Azure OpenAI services with seamless handling of memory/chat history, tools, and structured output.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Azure OpenAI Integration](#azure-openai-integration)
- [Advanced Usage](#advanced-usage)
- [Best Practices](#best-practices)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Installation

```bash
npm install kanuni-openai
```

## Quick Start

```typescript
import { Kanuni } from "kanuni";
import { AzureOpenAIChatCompletionsFormatter } from "kanuni-openai";
import { AzureOpenAI } from "openai";
import { z } from "zod";

// Create formatter
const formatter = new AzureOpenAIChatCompletionsFormatter();

// Build a Kanuni query
const query = Kanuni.newQuery<{ topic: string }>()
  .prompt(
    (p) =>
      p.paragraph`Explain ${"topic"} in simple terms.`
        .paragraph`Provide clear examples and avoid technical jargon.`
  )
  .memory((m) =>
    m.utterance("user", (data) => `I want to learn about ${data.topic}`)
  )
  .build({ topic: "machine learning" });

// Format for OpenAI API
const { messages, response_format, tools } = formatter.format(query);

// Use with Azure OpenAI
const client = new AzureOpenAI({
  apiKey: process.env.AZURE_OPENAI_API_KEY,
  endpoint: process.env.AZURE_OPENAI_ENDPOINT,
  apiVersion: "2024-10-21",
});

const response = await client.chat.completions.create({
  model: "gpt-4o",
  messages,
  ...(response_format && { response_format }),
  ...(tools && tools.length > 0 && { tools }),
});

console.log(response.choices[0].message.content);
```

## Core Concepts

The `AzureOpenAIChatCompletionsFormatter` transforms Kanuni queries into OpenAI-compatible parameters:

- **Message Structure**: Converts memory items to OpenAI message format
- **Role Mapping**: Maps roles to OpenAI's `user`, `assistant`, `system`, `developer`, `tool` message types
- **Tool Definitions**: Transforms Kanuni tools to OpenAI function calling format
- **Response Formats**: Converts JSON schemas to structured output format

### Kanuni Memory Items to Messages Conversion

1. Instructions → system/developer messages
2. Utterances → user/assistant messages
3. Tool calls → attached to assistant messages
4. Tool results → separate tool messages with call IDs

## API Reference

### AzureOpenAIChatCompletionsFormatter

#### Constructor

```typescript
new AzureOpenAIChatCompletionsFormatter<OutputType, ToolsType, Role>(config?)
```

#### Configuration Options

```typescript
interface AzureOpenAIChatCompletionsFormatterConfig {
  /**
   * Role for instructions/prompts.
   * Use "developer" for o1 models, "system" for others (default: "system")
   */
  instructionsRole?: "system" | "developer";

  /**
   * Custom function to format instructions from Kanuni queries
   * Default: Uses TextualMarkdownFormatter from Kanuni
   */
  instructionsFormatter?: (query: Query) => string;

  /**
   * Maps Kanuni roles to OpenAI message roles
   * Default: Identity mapping for 'user' and 'assistant'
   */
  roleMapper?: (sourceRole: Role, name?: string) => "user" | "assistant";
}
```

#### Methods

##### format(query, params?)

Converts a Kanuni query into OpenAI Chat Completions parameters.

**Parameters:**

- `query`: Kanuni query object
- `params`: Additional formatting parameters (currently unused)

**Returns:**

```typescript
{
  messages: ChatCompletionMessageParam[];
  response_format?: AutoParseableResponseFormat;
  tools?: ChatCompletionTool[];
}
```

## Examples

### Basic Text Query

```typescript
import { Kanuni } from "kanuni";
import { AzureOpenAIChatCompletionsFormatter } from "kanuni-openai";

const formatter = new AzureOpenAIChatCompletionsFormatter();

const query = Kanuni.newQuery<{ question: string }>()
  .prompt((p) => p.paragraph`Answer this question: ${"question"}`)
  .build({ question: "What is TypeScript?" });

const formatted = formatter.format(query);
console.log(formatted.messages);
// [
//   { role: "system", content: "Answer this question: What is TypeScript?" }
// ]
```

### JSON Structured Output

```typescript
const PersonSchema = z.object({
  name: z.string(),
  age: z.number(),
  skills: z.array(z.string()),
});

const query = Kanuni.newQuery<{ text: string }>()
  .prompt((p) => p.paragraph`Extract person info: ${"text"}`)
  .outputJson(PersonSchema, "person_extraction")
  .build({ text: "John Doe, 30, skilled in TypeScript and React" });

const response = await client.chat.completions.parse({
  model: "gpt-4o",
  ...formatter.format(query),
});

const person = response.choices[0].message.parsed; // Typed as PersonSchema
```

### Conversation Memory

```typescript
const query = Kanuni.newQuery<{ newMessage: string }>()
  .prompt((p) => p.paragraph`You are a helpful assistant.`)
  .memory((m) =>
    m
      .utterance("user", () => "Hello, my name is Alice")
      .utterance("assistant", () => "Hi Alice! Nice to meet you.")
      .utterance("user", (data) => data.newMessage)
  )
  .build({ newMessage: "What can you help me with?" });

const formatted = formatter.format(query);
console.log(formatted.messages);
// [
//   { role: "system", content: "You are a helpful assistant." },
//   { role: "user", content: "Hello, my name is Alice" },
//   { role: "assistant", content: "Hi Alice! Nice to meet you." },
//   { role: "user", content: "What can you help me with?" }
// ]
```

### Tool Usage

```typescript
type WeatherTool = Tool<"get_weather", { location: string; units?: string }>;

const tools: ToolRegistry<WeatherTool> = {
  get_weather: {
    name: "get_weather",
    description: "Get current weather for a location",
    parameters: {
      location: z.string().describe("City name or coordinates"),
      units: z.enum(["celsius", "fahrenheit"]).optional(),
    },
  },
};

const query = Kanuni.newQuery<{ request: string }, never, WeatherTool>()
  .prompt((p) => p.paragraph`Help with: ${"request"}`)
  .tools(tools)
  .memory((m) => m.utterance("user", (data) => data.request))
  .build({ request: "What's the weather in London?" });

const response = await client.chat.completions.create({
  model: "gpt-4o",
  ...formatter.format(query),
  tool_choice: "auto",
});
```

### Memory with Tool Results

```typescript
const query = Kanuni.newQuery<{}, never, WeatherTool>()
  .memory((m) =>
    m
      .utterance("user", () => "Weather in Tokyo?")
      .toolCall("get_weather", '{"location": "Tokyo"}', "call_123")
      .toolCallResult("call_123", "22°C, Sunny")
      .utterance("assistant", () => "Tokyo is sunny at 22°C")
  )
  .build({});
```

## OpenAI Integration

### Setup

```typescript
// Azure OpenAI
import { AzureOpenAI } from "openai";
const client = new AzureOpenAI({
  apiKey: process.env.AZURE_OPENAI_API_KEY,
  endpoint: process.env.AZURE_OPENAI_ENDPOINT,
  apiVersion: "2024-10-21",
});

// Regular OpenAI
import { OpenAI } from "openai";
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const formatter = new AzureOpenAIChatCompletionsFormatter();
```

### Model-Specific Configuration

```typescript
// o1 models use "developer" role for instructions
const o1Formatter = new AzureOpenAIChatCompletionsFormatter({
  instructionsRole: "developer",
});

// Other models use "system" role (default)
const standardFormatter = new AzureOpenAIChatCompletionsFormatter();
```

### Complete Example

```typescript
async function processWithKanuni(userInput: string) {
  const query = Kanuni.newQuery<{ input: string }>()
    .prompt((p) => p.paragraph`You are an expert assistant. Input: ${"input"}`)
    .outputJson(
      z.object({
        response: z.string(),
        confidence: z.number().min(0).max(1),
      }),
      "assistant_response"
    )
    .build({ input: userInput });

  const response = await client.chat.completions.parse({
    model: "gpt-4o",
    ...formatter.format(query),
    temperature: 0.7,
  });

  return response.choices[0].message.parsed;
}
```

## Advanced Usage

### Custom Instructions Formatting

```typescript
import { TextualMarkdownFormatter } from "kanuni";

const customFormatter = new AzureOpenAIChatCompletionsFormatter({
  instructionsFormatter: (query) => {
    // Custom logic to format instructions
    const baseInstructions = new TextualMarkdownFormatter().format(query);
    return `CUSTOM SYSTEM PROMPT:\n\n${baseInstructions}\n\nAlways be concise.`;
  },
});
```

### Custom Role Mapping

```typescript
type CustomRole = "user" | "assistant" | "moderator";

const formatter = new AzureOpenAIChatCompletionsFormatter<
  any,
  never,
  CustomRole
>({
  roleMapper: (sourceRole, name) => {
    if (sourceRole === "moderator") return "assistant";
    return sourceRole; // user, assistant pass through
  },
});
```

### Conversation Continuation

```typescript
// Start conversation
const initialQuery = Kanuni.newQuery<{ question: string }>()
  .prompt((p) => p.paragraph`You are a helpful tutor.`)
  .memory((m) => m.utterance("user", (data) => data.question))
  .build({ question: "What is recursion?" });

let memory = Kanuni.extractMemoryFromQuery(initialQuery);

// Continue conversation
const followUpQuery = Kanuni.newQuery<{ followUp: string }>()
  .prompt((p) => p.paragraph`Continue helping the student.`)
  .memory((m) =>
    m
      .append(memory?.contents || [])
      .utterance("assistant", () => "Recursion is a programming technique...")
      .utterance("user", (data) => data.followUp)
  )
  .build({ followUp: "Can you show me an example?" });

// Keep building conversation memory...
```

### Error Handling

```typescript
import { AzureOpenAI } from "openai";

const client = new AzureOpenAI(/* config */);
const formatter = new AzureOpenAIChatCompletionsFormatter();

try {
  const formatted = formatter.format(query);

  const response = await client.chat.completions.parse({
    model: "gpt-4o",
    messages: formatted.messages,
    response_format: formatted.response_format,
  });

  if (response.choices[0].message.parsed) {
    return response.choices[0].message.parsed;
  } else {
    console.error(
      "Failed to parse response:",
      response.choices[0].message.refusal
    );
  }
} catch (error) {
  if (error instanceof Error) {
    if (error.message.includes("Unknown role")) {
      console.error("Role mapping error:", error.message);
    } else if (error.message.includes("output type")) {
      console.error("Output schema error:", error.message);
    } else {
      console.error("Formatting error:", error.message);
    }
  }
  throw error;
}
```

## Troubleshooting

### Common Issues & Solutions

```typescript
// 1. "Unknown role" errors - map custom roles
const formatter = new AzureOpenAIChatCompletionsFormatter({
  roleMapper: (role, name) => {
    if (["moderator", "admin"].includes(role)) return "assistant";
    if (["customer", "guest"].includes(role)) return "user";
    return role as "user" | "assistant";
  },
});

// 2. JSON schema failures - use simple, strict schemas
const schema = z
  .object({
    result: z.string().describe("Clear result description"),
    metadata: z.record(z.string()).optional(),
  })
  .strict();

// 3. Memory ordering - keep tool calls with results
const query = Kanuni.newQuery().memory((m) =>
  m
    .utterance("user", () => "Question")
    .toolCall("tool", "args", "id") // Tool call
    .toolCallResult("id", "result") // Immediate result
    .utterance("assistant", () => "Response")
);
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. When contributing:

1. Follow the existing code style
2. Add tests for new functionality
3. Update documentation as needed
4. Ensure TypeScript types are properly defined

### Development Setup

```bash
git clone <repository-url>
cd kanuni-openai
npm install
npm run build
npm test
```

## License

MIT

---

For more information about Kanuni's query building capabilities, see the [Kanuni documentation](https://www.npmjs.com/package/kanuni).
