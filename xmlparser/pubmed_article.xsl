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

    <html>
      <body>

        <div class="metadata">
          <p>Excerpt from:
          <xsl:apply-templates select="//*[name()='title-group']"/></p>
          <p>Authors: <xsl:apply-templates select="//*[name()='contrib']/*[name()='name']"/></p>
          <p>DOI: <xsl:value-of select="//*[name()='article-id' and @pub-id-type='doi']"/></p>
        </div>
        
        <div class="abstract">
          <h3>Abstract:</h3>
          <xsl:apply-templates select="//*[name()='abstract']"/>
        </div>

        <div class="body">
          <xsl:apply-templates select="//*[name()='body']/*"/>
        </div>
      </body>
    </html>
  </xsl:template>
  
  <xsl:template match="//*">
    <xsl:copy>
      <xsl:copy-of select="@*"/>
      <xsl:apply-templates/>
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

  <xsl:template match="//*[name()='body']//*[name()='title']">
    <h3><xsl:apply-templates/></h3>
  </xsl:template>

</xsl:stylesheet>
