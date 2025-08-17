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

/**
 * The roles that this formatter supports for utterences in Kanuni.
 * These are also the ones that are the user and assistant types for
 * OpenAI messages.
 */
const SUPPORTED_ROLES = ['user', 'assistant'] as const;
/**
 * Roles that this formatter supports for utterances in Kanuni.
 */
export type SupportedRoles = (typeof SUPPORTED_ROLES)[number];

// The following type definition checks to make sure that we continue to stay
// in alignment with the openai library.
type ChatCompletionUtteranceRoles = Extract<ChatCompletionRole, SupportedRoles>;

/**
 * Supported OpenAI message types for instructions.
 */
type SupportedInstructionsRoles= 'developer' | 'system';

// The following type definition checks to make sure that we continue to stay
// in alignment with the openai library.
type ChatCompletionInstructionsRoles = Extract<ChatCompletionRole, SupportedInstructionsRoles>;

/**
 * Type of the function that's used for formatting the instructions, which is
 * the "prompt" part of the Kanuni query.
 */
export type InstructionsFormatterFunction<
  OutputSchema extends Record<string, any> | string,
  Role extends string = ChatCompletionUtteranceRoles,
  ToolsType extends Tool<any, any> = never
> = (query: Query<OutputSchema, Role, ToolsType>) => string;

/**
 * Type of the function that is used for mapping roles used in Kanuni query
 * to the supported roles in this formatter. See {@see SupportedRoles}.
 */
export type RoleMapperFunction<SourceRole extends string> =
  (sourceRole: SourceRole, name?: string) => ChatCompletionUtteranceRoles;

/**
 * The type of the configuration object this formatter accepts.
 */
export type AzureOpenAIChatCompletionsFormatterConfig<
  OutputSchema extends Record<string, any> | string,
  Role extends string = ChatCompletionUtteranceRoles,
  ToolsType extends Tool<any, any> = never
> = {
  /**
   * Post o1 models use "developer", others use "system". Default is "system".
   * Check which role your model uses for instructions.
   */
  instructionsRole?: ChatCompletionInstructionsRoles;

  // TODO: add model and version params so that the formatter can adapt to
  // different OpenAI models and versions.

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

  /**
   * Maps the role used in the Kanuni query to the supported roles in this formatter.
   * 
   * The default is an identity function that maps the kanuni role to the same role name.
   * 
   * @param sourceRole The role used in the Kanuni query.
   * @param name The name of the role, if any.
   * @returns The mapped role that is supported by this formatter.
   */
  roleMapper?: RoleMapperFunction<Role>;
}

/**
 * The default message type to use for instructions.
 */
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
  /**
   * The OpenAI message type to use for the instructions.
   */
  private instructionsRole: ChatCompletionInstructionsRoles;

  /**
   * The formatter to use for generating the instructions.
   * Default is TextualMarkdownFormatter from Kanuni.
   */
  private instructionsFormatter: InstructionsFormatterFunction<OutputType, Role, ToolsType>;

  /**
   * The role mapper to use for mapping Kanuni query roles to the
   * supported roles in this formatter.
   * Default is the identity mapping function.
   */
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

  /**
   * The identity role mapping function used as a default value for the
   * role mapper in this formatter.
   * 
   * @param sourceRole Role used in the query.
   * @returns Message type to use for the role in OpenAI messages.
   */
  private identityRoleMapper(sourceRole: Role): SupportedRoles {
    const mappedRole = SUPPORTED_ROLES.find(role => role === sourceRole);
    if (mappedRole !== undefined) {
      return mappedRole;
    }
    throw new Error(`Unknown role: ${sourceRole}`);
  }

  /**
   * Format the given query into parameters for OpenAI chat completions API.
   * 
   * @param query Kanuni query context.
   * @param _params Currently unused in this formatter. Normally additional
   * parameters for the formatter.
   * @returns Parameters that can be passed to openai.chat.completions
   * functions.
   */
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

  /**
   * Format the tools from the given query into the format that is acceptable
   * by OpenAI chat completions API.
   * 
   * @param query Kanuni query with tools.
   * @returns Tool definitions that can be used with OpenAI chat completions
   * API.
   */
  private formatTools(
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
        parameters:
          // These parameters are taken from the method used within openai
          // library for formatting zod definitions to json schemas.
          zodToJsonSchema(z.strictObject(tool.parameters), {
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

  /**
   * Format the memory items from the given query into the format that is
   * acceptable by OpenAI chat completions API.
   * 
   * @param query Kanuni query with memory items.
   * @returns Array of messages that can be used with OpenAI chat completions.
   */
  private formatMemoryItems(
    query: Query<OutputType, Role, ToolsType>,
  ): AzureOpenAIChatCompletionsFormatterResult['messages'] {
    // A note about what this method does:
    // The tool calls are represented as individual memory items in the Kanuni's
    // memory representation. However, OpenAI represents tool calls within the
    // assistant message. Hence, this method regroups Kanuni memory items before
    // converting them to OpenAI messages.

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
    ];

    return memoryItems;
  }

  /**
   * Convert the given tool call to the format acceptable by openai library.
   * 
   * @param toolCall The tool call in Kanuni format.
   * @returns Tool call in the format that is acceptable by OpenAI chat
   * completions API.
   */
  private formatToolCall(toolCall: ToolCall<ToolsType["name"]>): ChatCompletionMessageToolCall {
    return {
      id: toolCall.toolCallId,
      type: 'function',
      function: {
        name: toolCall.toolName,
        arguments: toolCall.arguments,
      },
    };
  }

  /**
   * Format the json output schema from the given query into the format
   * that is acceptable by OpenAI chat completions API.
   * 
   * @param query Kanuni query with output schema defined.
   * @returns Parsable response format that can be used with OpenAI chat
   * completions API, especially the `parse` method.
   */
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
