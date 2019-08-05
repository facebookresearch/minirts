# Installation

Install Node.js if node version < 8.
On Mac, run
```
brew install node
```
On Linux, run
```
sudo apt install nodejs
sudo apt install npm
```
Or get installer from [web](https://nodejs.org/en/download/).

Install npm packages. Run in project root folder:
```
yarn
```

# Run
Generate the json file list:
```
python gen_file_list.py
```

Start server, and auto rebuild on change:
```
npm start
```
This should automatically launch a browser tab for the visualization program.

In the `select game` text bar, we can use the scroll down menu to select a game for
replay, or type in the name of a particular replay to search. We can also manually type
in the specific tick/entry number in the text bar below to go to a particular tick, in
order to investigate what is happening at that moment.

# Build
```
npm run build
```
This will minify and put a standalone version in "build" folder