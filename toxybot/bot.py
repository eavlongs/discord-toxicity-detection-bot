import os
import discord
import random
from dotenv import load_dotenv
from toxic import predict_toxicity, OUTPUT_LABEL

# Load the token from the .env file
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

# Command prefix
COMMAND_PREFIX = '/toxme'

# File to store followed channels
CHANNELS_FILE = 'followed_channels.txt'

# Function to load followed channels from file


def load_followed_channels():
    followed_channels = {}
    try:
        with open(CHANNELS_FILE, 'r') as file:
            for line in file:
                server_id, channel_id = map(int, line.strip().split(','))
                if server_id not in followed_channels:
                    followed_channels[server_id] = set()
                followed_channels[server_id].add(channel_id)
    except FileNotFoundError:
        pass
    return followed_channels

# Function to save followed channels to file


def save_followed_channels(followed_channels):
    with open(CHANNELS_FILE, 'w') as file:
        for server_id, channels in followed_channels.items():
            for channel_id in channels:
                file.write(f'{server_id},{channel_id}\n')


async def detect_toxicity(message: discord.Message):
    # label 0 is non-toxic, label 1 is toxic
    if predict_toxicity(message.content) == OUTPUT_LABEL[1]:
        print(f'Toxic message detected: {message.content}')
        
        # delete the message and warn the user
        await message.delete()
        await message.channel.send(f'{message.author.mention}, your message was deleted because it was a toxic message.')

# Create an instance of a client
intents = discord.Intents.default()
intents.message_content = True  # Enables reading message content
client = discord.Client(intents=intents)

followed_channels = load_followed_channels()  # Load followed channels from file


@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')


@client.event
async def on_message(message: discord.Message):
    # Ignore messages from the bot itself
    if message.author == client.user:
        return

    IS_COMMAND_PREFIX = message.content.startswith(COMMAND_PREFIX)
    IS_MESSAGE_FROM_FOLLOWED_CHANNEL = message.guild.id in followed_channels and message.channel.id in followed_channels[
        message.guild.id]

    # Check if the message starts with the command prefix
    if IS_COMMAND_PREFIX:
        parts = message.content.split()
        if len(parts) == 1:
            # Display command hints
            command_hints = [
                f'**{COMMAND_PREFIX} follow <channel_name>**: Follows the specified channel',
                f'**{COMMAND_PREFIX} unfollow <channel_name>**: Unfollows the specified channel',
                f'**{COMMAND_PREFIX} list**: Lists all followed channels',
                f'**{COMMAND_PREFIX} help**: Displays command hints'
            ]
            await message.channel.send('\n'.join(command_hints))
        elif len(parts) > 1:
            command = parts[1]
            if command == 'follow' and len(parts) == 3:
                # Follow a channel
                channel_name = parts[2]
                channel = discord.utils.get(
                    message.guild.channels, name=channel_name)
                if channel:
                    if message.guild.id not in followed_channels:
                        followed_channels[message.guild.id] = set()
                    followed_channels[message.guild.id].add(channel.id)
                    # Save followed channels to file
                    save_followed_channels(followed_channels)
                    await message.channel.send(f'Now following <#{channel.id}>')
                else:
                    await message.channel.send(f'Channel #{channel_name} not found')
            elif command == 'unfollow' and len(parts) == 3:
                # Unfollow a channel
                channel_name = parts[2]
                channel = discord.utils.get(
                    message.guild.channels, name=channel_name)
                if channel and channel.id in followed_channels.get(message.guild.id, set()):
                    followed_channels[message.guild.id].remove(channel.id)
                    # Save followed channels to file
                    save_followed_channels(followed_channels)
                    await message.channel.send(f'Stopped following <#{channel.id}>')
                elif not channel:
                    await message.channel.send(f'Channel #{channel_name} not found')
                else:
                    await message.channel.send(f'Not following #{channel_name}')
            elif command == 'list':
                # List followed channels
                if message.guild.id in followed_channels:
                    channel_list = [
                        f'<#{channel_id}>' for channel_id in followed_channels[message.guild.id]]
                    await message.channel.send(f'Followed channels:\n{" ".join(channel_list)}')
                else:
                    await message.channel.send('No channels being followed.')
            elif command == 'help':
                # Display command hints
                command_hints = [
                    f'**{COMMAND_PREFIX} follow <channel_name>**: Follows the specified channel',
                    f'**{COMMAND_PREFIX} unfollow <channel_name>**: Unfollows the specified channel',
                    f'**{COMMAND_PREFIX} list**: Lists all followed channels',
                    f'**{COMMAND_PREFIX} help**: Displays command hints'
                ]
                await message.channel.send('\n'.join(command_hints))
            else:
                await message.channel.send('Invalid command usage.')
        else:
            await message.channel.send('Invalid command usage.')

    # Only reply if the message is in a followed channel
    elif IS_MESSAGE_FROM_FOLLOWED_CHANNEL:
        await detect_toxicity(message)
        # Mention the user who sent the message and include their message content
        # channel_name = message.channel.name
        # await message.channel.send(f'{message.content} - {message.author.mention}, in <#{message.channel.id}>')

# Run the bot
client.run(TOKEN)
