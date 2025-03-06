# Hexoskin WAV Analyzer Documentation

This directory contains the files for the GitHub Pages site of the Hexoskin WAV Analyzer project.

## Structure

- `index.md`: Main page of the site
- `_config.yml`: Configuration file for GitHub Pages
- `assets/`: Directory for assets like images and CSS files

## Local Development

To test the site locally, you need to have Jekyll installed. Follow these steps:

1. Install Ruby and Jekyll: https://jekyllrb.com/docs/installation/
2. Navigate to the `docs` directory
3. Run `bundle install`
4. Run `bundle exec jekyll serve`
5. Open your browser and go to `http://localhost:4000`

## Adding Content

To add new pages to the site:

1. Create a new Markdown file in the `docs` directory
2. Add the front matter at the top of the file:
   ```
   ---
   layout: default
   title: Your Page Title
   ---
   ```
3. Add your content in Markdown format
4. Link to the new page from the main page or other pages

## Deployment

The site is automatically deployed to GitHub Pages when changes are pushed to the main branch. 