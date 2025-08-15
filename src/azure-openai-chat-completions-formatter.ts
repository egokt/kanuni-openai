import type {
  Formatter,
  Query,
  Tool,
  ToolCall,
} from "kanuni";
import { TextualMarkdownFormatter } from "kanuni";
import { AzureOpenAI } from "openai";
import { zodToJsonSchema } from "openai/_vendor/zod-to-json-schema/zodToJsonSchema.mjs";
import { zodResponseFormat } from "openai/helpers/zod.js";
import { AutoParseableResponseFormat } from "openai/lib/parser.js";
import {
  ChatCompletionAssistantMessageParam,
  ChatCompletionMessageToolCall,
  ChatCompletionRole,
  ChatCompletionTool,
  ChatCompletionToolMessageParam,
  ChatCompletionUserMessageParam,
} from "openai/resources";
import { ZodType } from "zod";
import z from "zod/v3";

const SUPPORTED_ROLES = ['user', 'assistant'] as const;
export type SupportedRoles = (typeof SUPPORTED_ROLES)[number];

// The following checks to make sure that we continue to stay in alignment with
// the openai library
type ChatCompletionUtteranceRoles = Extract<ChatCompletionRole, SupportedRoles>;

type SupportedInstructionsRoles= 'developer' | 'system';
type ChatCompletionInstructionsRoles = Extract<ChatCompletionRole, SupportedInstructionsRoles>;

export type InstructionsFormatterFunction<
  OutputSchema extends Record<string, any> | string,
  Role extends string = ChatCompletionUtteranceRoles,
  ToolsType extends Tool<any, any> = never
> = (query: Query<OutputSchema, Role, ToolsType>) => string;

export type RoleMapperFunction<SourceRole extends string> =
  (sourceRole: SourceRole, name?: string) => ChatCompletionUtteranceRoles;

export type AzureOpenAIChatCompletionsFormatterConfig<
  OutputSchema extends Record<string, any> | string,
  Role extends string = ChatCompletionUtteranceRoles,
  ToolsType extends Tool<any, any> = never
> = {
  /**
   * Post o1 models use "developer", others use "system". Default is "system". Check which role your model uses for instructions.
   */
  instructionsRole?: ChatCompletionInstructionsRoles;

  // TODO: add model and version params so that the formatter can adapt to different OpenAI models and versions.

  /**
   * Formats the instructions for the query.
   * This is the part of the query without memory section, output, or tools.
   * 
   * The default is the format method of the TextualMarkdownFormatter in Kanuni.
   * 
   * @param query The query to format instructions for.
   * @returns The formatted instructions.
   */
  instructionsFormatter?: InstructionsFormatterFunction<OutputSchema, Role, ToolsType>;

  roleMapper?: RoleMapperFunction<Role>;
}

const DEFAULT_INSTRUCTIONS_ROLE: ChatCompletionInstructionsRoles = "system";

type AzureOpenAIChatCompletionsFormatterParams = {};

type AzureOpenAIChatCompletionsFormatterResult =
  Pick<Parameters<
    AzureOpenAI['chat']['completions']['parse']>[0],
    'messages' | 'response_format' | 'tools'
  >;

export class AzureOpenAIChatCompletionsFormatter<
  OutputType extends Record<string, any> | string,
  ToolsType extends Tool<any, any> = never,
  Role extends string = ChatCompletionUtteranceRoles,
> implements Formatter<
    AzureOpenAIChatCompletionsFormatterParams,
    AzureOpenAIChatCompletionsFormatterResult,
    OutputType,
    Role,
    ToolsType
  >
{
  private instructionsRole: ChatCompletionInstructionsRoles;
  private instructionsFormatter: InstructionsFormatterFunction<OutputType, Role, ToolsType>;
  private roleMapper: RoleMapperFunction<Role>;

  constructor(
    {
      instructionsRole = DEFAULT_INSTRUCTIONS_ROLE,
      instructionsFormatter = (query: Query<OutputType, Role, ToolsType>) => new TextualMarkdownFormatter<OutputType, Role, ToolsType>().format(query),
      roleMapper = this.identityRoleMapper,
    }: AzureOpenAIChatCompletionsFormatterConfig<OutputType, Role, ToolsType> = {},
  ) {
    this.instructionsRole = instructionsRole;
    this.instructionsFormatter = instructionsFormatter;
    this.roleMapper = roleMapper;
  }

  private identityRoleMapper(sourceRole: Role): SupportedRoles {
    const mappedRole = SUPPORTED_ROLES.find(role => role === sourceRole);
    if (mappedRole !== undefined) {
      return mappedRole;
    }
    throw new Error(`Unknown role: ${sourceRole}`);
  }

  format(
    query: Query<OutputType, Role, ToolsType>,
    _params: AzureOpenAIChatCompletionsFormatterParams = {},
  ): AzureOpenAIChatCompletionsFormatterResult {
    const memoryItems = this.formatMemoryItems(query);
    const tools = this.formatTools(query);
    let responseJsonSchema;
    switch (query.output.type) {
      case 'output-text':
        // the next line is noop: it is here for clarity
        responseJsonSchema = undefined;
        break;
      case 'output-json':
        responseJsonSchema = this.formatJsonSchema(query);
        break;
      default:
        throw new Error(`Unknown output type: ${(query.output as { type: string; }).type}`);
    }

    return {
      messages: memoryItems,
      ...(responseJsonSchema === undefined ? {} : { response_format: responseJsonSchema }),
      ...(tools === undefined ? {} : { tools }),
    };
  }

  formatTools(
    query: Query<OutputType, Role, ToolsType>
  ): AzureOpenAIChatCompletionsFormatterResult['tools'] {
    const toolRegistry = query.tools;

    if (toolRegistry === undefined || Object.keys(toolRegistry).length === 0) {
      return [];
    }

    return Object.values<ToolsType>(toolRegistry).map(tool => ({
      type: 'function',
      function: {
        name: tool.name,
        description: tool.description,
        parameters: zodToJsonSchema(z.strictObject(tool.parameters), {
          openaiStrictMode: true,
          name: tool.name,
          nameStrategy: 'duplicate-ref',
          $refStrategy: 'extract-to-root',
          nullableStrategy: 'property',
        }),
        strict: true,
      }
    } as ChatCompletionTool));
  }

  // Warn: This method only supports utterances in the memory section.
  // TODO: Extend this to support other types of memory items, i.e. tools, when they are implemented.
  private formatMemoryItems(
    query: Query<OutputType, Role, ToolsType>,
  ): AzureOpenAIChatCompletionsFormatterResult['messages'] {
    const instructionsRole = this.instructionsRole;
    const instructions = this.instructionsFormatter(query);

    let lastGroup:
      | {
        // this type is for user utterance

        type: 'user',
        utteranceItem: Extract<NonNullable<typeof query.memory>['contents'][number], { type: 'utterance'; }>;
      }
      | {
        type: 'assistant',

        // utterance may not be present if the llm chose to call tools without any accompanying text output
        utteranceItem?: Extract<NonNullable<typeof query.memory>['contents'][number], { type: 'utterance'; }>;

        toolCalls: Extract<NonNullable<typeof query.memory>['contents'][number], { type: 'tool-call'; }>[];
      }
      | {
        type: 'tool-call-result',
        toolCallResults: Extract<NonNullable<typeof query.memory>['contents'][number], { type: 'tool-call-result'; }>[];
      }
      | null;
    const regroupedMemoryItems: (NonNullable<typeof lastGroup>)[] = [];
    lastGroup = null;
    for (const memoryItem of query.memory?.contents || []) {
      switch (memoryItem.type) {
        case 'utterance':
          if (lastGroup !== null) {
            if (lastGroup.type === 'assistant' && lastGroup.utteranceItem === undefined) {
              // this is a bit interesting, but handle anyways:
              // there are tool calls, then an assistance utterance in the memory
              // combine them together as a message for the openai messages in prompt context
              lastGroup.utteranceItem = memoryItem;
            } else {
              regroupedMemoryItems.push(lastGroup);
              switch (this.roleMapper(memoryItem.role, memoryItem.name)) {
                case 'user':
                  lastGroup = {
                    type: 'user',
                    utteranceItem: memoryItem,
                  };
                  break;
                case 'assistant':
                  lastGroup = {
                    type: 'assistant',
                    utteranceItem: memoryItem,
                    toolCalls: [],
                  };
                  break;
                default:
                  throw new Error(`Only user and assistant utterances are supported by AzureOpenAIChatCompletionsJsonFormatter`);
              }
            }
          } else {
            switch (this.roleMapper(memoryItem.role, memoryItem.name)) {
              case 'user':
                lastGroup = {
                  type: 'user',
                  utteranceItem: memoryItem,
                };
                break;
              case 'assistant':
                lastGroup = {
                  type: 'assistant',
                  utteranceItem: memoryItem,
                  toolCalls: [],
                };
                break;
              default:
                throw new Error(`Only user and assistant utterances are supported by AzureOpenAIChatCompletionsJsonFormatter`);
            }
          }
          break;
        case 'tool-call':
          if (lastGroup === null) {
            // this is a weird case: the memory starts with the assistant
            // making a tool call
            lastGroup = {
              type: 'assistant',
              toolCalls: [memoryItem],
            };
          } else {
            if (lastGroup.type === 'assistant') {
              lastGroup.toolCalls.push(memoryItem);
            } else {
              // we assume the last group not being an assistant utterance or
              // tool call means the assistant did not output any text and
              // its answer is containing only tool cals (i.e. it is starting a
              // new answer).
              regroupedMemoryItems.push(lastGroup);
              lastGroup = {
                type: 'assistant',
                toolCalls: [memoryItem],
              };
            }
          }
          break;
        case 'tool-call-result':
          if (lastGroup === null) {
            throw new Error('Memory starts with a tool call result without ' +
              `its corresponding call. Tool call id: ${memoryItem.toolCallId}`);
          } else {
            if (lastGroup.type === 'tool-call-result') {
              lastGroup.toolCallResults.push(memoryItem);
            } else {
              regroupedMemoryItems.push(lastGroup);
              lastGroup = {
                type: 'tool-call-result',
                toolCallResults: [memoryItem],
              };
            }
          }
          break;
        default:
          throw new Error(`Unexpected memory type: ${(memoryItem as { type: string; }).type}`)
      }
    }
    if (lastGroup !== null) {
      regroupedMemoryItems.push(lastGroup);
    }

    const memoryItems = [
      {
        role: instructionsRole,
        content: instructions,
      },
      ...regroupedMemoryItems.map(item => {
        switch (item.type) {
          case 'user':
            return {
              role: 'user',
              content: item.utteranceItem.contents,
              ...(item.utteranceItem.name !== undefined
                ? { name: item.utteranceItem.name }
                : {}),
            } as ChatCompletionUserMessageParam;
          case 'assistant':
            return {
              role: 'assistant',
              ...(item.utteranceItem !== undefined ? { content: item.utteranceItem.contents } : {}),
              ...(item.toolCalls.length > 0 ? { tool_calls: item.toolCalls.map(toolCall => this.formatToolCall(toolCall)) } : {})
            } as ChatCompletionAssistantMessageParam;
          case 'tool-call-result':
            return item.toolCallResults.map(toolCallResult => ({
              role: 'tool',
              content: toolCallResult.result === null ? 'Success' : toolCallResult.result,
              tool_call_id: toolCallResult.toolCallId,
            } as ChatCompletionToolMessageParam));
          default:
            throw new Error(`Internal error in AzureOpenAIChatCompletionsJsonFormatter.`);
        }
      }).flat(),
      // ...(query.memory?.contents || []).map((item) => {
      //   switch (item.type) {
      //     case "utterance":
      //       const role = this.roleMapper(item.role, item.name);
      //       if (!UTTERANCE_ROLES[role]) {
      //         throw new Error(`Role mapping for '${item.role}' resulted in unknown utterance role: ${role}`);
      //       }
      //       return {
      //         role: role as UtteranceRole, // this type casting is safe due to the conditional above
      //         ...(item.name !== undefined && item.name !== '' ? { name: item.name } : {}),
      //         content: item.contents,
      //       };
      //     case 'tool-call':
      //       return {

      //       };
      //       break;
      //     case 'tool-call-result':
      //       break;
      //     default:
      //       throw new Error(`Unknown memory item type: ${item.type}`);
      //   }
      // }),
    ];

    return memoryItems;
  }

  /**
   * Convert the given tool call to the format acceptable by openai library.
   * 
   * @param toolCall The tool call in Kanuni format.
   */
  formatToolCall(toolCall: ToolCall<ToolsType["name"]>): ChatCompletionMessageToolCall {
    return {
      id: toolCall.toolCallId,
      type: 'function',
      function: {
        name: toolCall.toolName,
        arguments: toolCall.arguments,
      },
    };
  }

  private formatJsonSchema(
    query: Query<any, any, any>,
  ): AutoParseableResponseFormat<OutputType> {
    // This formatter is designed to work with queries that have a json output.
    // If the query does not have output schema set, throw an error.
    const output = query.output;
    if (output?.type !== 'output-json') {
      throw new Error(
        `Query output type must be 'output-json', but ` +
          output !== undefined
          ? `got '${(output as { type: string }).type}'`
          : 'output is left as default (text)'
      );
    }

    return zodResponseFormat<ZodType<OutputType>>(
      output.schema,
      output.schemaName,
    );
  }
}
