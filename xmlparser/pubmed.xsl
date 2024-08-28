<?xml version="1.0" encoding="UTF-8" ?>
<xsl:stylesheet version="1.0"
    	xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
		xmlns:ali="http://www.niso.org/schemas/ali/1.0/"
		xmlns:xlink="http://www.w3.org/1999/xlink" 
		xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
		xmlns="https://jats.nlm.nih.gov/ns/archiving/1.3/">
  <xsl:output method="html"
    	      encoding="UTF-8"
    	      indent="yes" />

  <xsl:template match="/">
    <xsl:apply-templates select="chunk"/>
    <xsl:apply-templates select="//*[name()='article']"/>
  </xsl:template>

  <xsl:template match="chunk">
    <xsl:copy>

      <div class="metadata">
        <p>Excerpt from:
        <strong>
          <xsl:apply-templates select="//*[name()='title-group']/*[name()='article-title']"/>
        </strong>
        </p>
        <p>Authors: <xsl:apply-templates select="//*[name()='contrib']/*[name()='name']"/></p>
        <p>DOI: <xsl:value-of select="//*[name()='article-id' and @pub-id-type='doi']"/></p>
      </div>

      <div class="chunk-body">
        <xsl:apply-templates select="chunk-body"/>
      </div>
      
    </xsl:copy>
  </xsl:template>

  <xsl:template match="//*[name()='article']">

    <html>
      <body>

        <div class="metadata">
          <h2><xsl:apply-templates select="//*[name()='title-group']/*[name()='article-title']"/></h2>
          <p>Authors: <xsl:apply-templates select="//*[name()='contrib']/*[name()='name']"/></p>
          <p>DOI: <xsl:value-of select="//*[name()='article-id' and @pub-id-type='doi']"/></p>
        </div>
        
        <xsl:apply-templates select="//*[name()='abstract']"/>

        <div class="article-body">
          <xsl:apply-templates select="//*[name()='body']/*"/>
        </div>
      </body>
    </html>
  </xsl:template>
  
  <xsl:template match="//*[name()='title-group']/*[name()='article-title']">
    <xsl:copy-of select="@*|node()"/>
  </xsl:template>

  <xsl:template match="//*[name()='abstract']">
    <div class="abstract">
      <h2>Abstract:</h2>
      <xsl:apply-templates/>
    </div>
  </xsl:template>

  <xsl:template match="//*[name()='fig']">
    <xsl:copy>
      <xsl:copy-of select="@*|node()"/>
    </xsl:copy>
  </xsl:template>
  
  <xsl:template match="//*[name()='p' or name()='table-wrap']">
    <xsl:copy>
      <xsl:copy-of select="@*|node()"/>
    </xsl:copy>
  </xsl:template>


  <xsl:template match="//*[name()='surname']">
    <xsl:copy>
      <xsl:copy-of select="@*"/>
      <xsl:apply-templates/>
      <xsl:if test="not(position()=last())">, </xsl:if>
    </xsl:copy>
  </xsl:template>
  
  <xsl:template match="//*[name()='name']">
    <xsl:copy>
      <xsl:copy-of select="@*"/>
      <xsl:apply-templates/>
      <xsl:if test="not(position()=last())"> - </xsl:if>
    </xsl:copy>
  </xsl:template>

  <xsl:template match="//*[name()='sec']/*[name()='title']">
    <h3><xsl:apply-templates/></h3>
  </xsl:template>

  <xsl:template match="//*[name()='sec']/*[name()='sec']/*[name()='title']">
    <h4><xsl:apply-templates/></h4>
  </xsl:template>

  <xsl:template match="//*[name()='sec']/*[name()='sec']/*[name()='sec']/*[name()='title']">
    <h5><xsl:apply-templates/></h5>
  </xsl:template>

</xsl:stylesheet>
